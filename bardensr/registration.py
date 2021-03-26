from . import kernels
import tensorflow as tf
from . import misc
import scipy as sp
import scipy.ndimage
import numpy as np

def translate_frames(X,t,pad,method='hermite'):
    '''
    batched version of hermite_translation_on_grid
    '''
    
    if method =='hermite':
        newXs = [kernels.hermite_translation_on_grid(X[f],t[f],pad) for f in range(X.shape[0])]
    elif method =='linear':
        newXs = [kernels.linear_translation_on_grid(X[f],t[f],pad) for f in range(X.shape[0])]
    else:
        raise NotImplementedError(method)

    return tf.stack(newXs,axis=0)

def calc_loss_and_grad(X,code,t,pad,interpolation_method='hermite'):
    '''
    X -- F x M0 x M1 x M2
    code -- F x J
    t -- F x 3
    pad -- 3
    '''
    
    with tf.GradientTape() as tape:
        tape.watch(t)
        newX = translate_frames(X,t,pad,method=interpolation_method)

        dots=tf.einsum('fj,fabc->abcj',code,newX)
        mx = tf.reduce_max(dots,axis=-1)
        loss=-tf.reduce_sum(mx**2)
        
    grad=tape.gradient(loss,t)
    return loss,grad


def preprocess_for_registration(mini,bgradius=(10,10,10),mnmxnorm=True):
    # basic bg subtraction
    if bgradius is not None:
        mini=mini-sp.ndimage.gaussian_filter(mini,(0,0,*bgradius))
        mini=np.clip(mini,0,np.inf)

    # and normalization...
    if mnmxnorm:
        mini=mini-mini.min(axis=(2,3,4),keepdims=True)
        mini=mini/mini.max(axis=(2,3,4),keepdims=True)

    return mini

def apply_registration(mini,t,zero_padding,interpolation_method='hermite'):
    '''
    Shifts mini by translations t, using some zero_padding to deal
    with translations out of frame

    For example, if t=0, then we would have that

    result[0,0,m0,m1,m2] <-- mini[0,0,m0+1,m1+1,m2+1]

    '''
    R,C=mini.shape[:2]

    # pad the data
    ZERO_PAD=[
        (0,0),
        (zero_padding[0],zero_padding[0]),
        (zero_padding[1],zero_padding[1]),
        (zero_padding[2],zero_padding[2])
    ]
    minitf = tf.identity(np.reshape(mini,(R*C,)+mini.shape[2:]).astype(np.float32))
    minitf=tf.pad(minitf,ZERO_PAD)

    # store the translation pad for tensorflow to use
    pad=tf.convert_to_tensor(zero_padding,dtype=np.int32)

    trans=translate_frames(minitf,t,pad,method=interpolation_method).numpy()
    return np.reshape(trans,(R,C)+trans.shape[1:])

def register(mini,codebook,zero_padding=(20,20,20),
                    use_tqdm_notebook=False,niter=100,initial_step=.1,momentum=.9,
                    interpolation_method='hermite'):
    R,C=mini.shape[:2]

    # pad the data
    ZERO_PAD=[
        (0,0),
        (zero_padding[0],zero_padding[0]),
        (zero_padding[1],zero_padding[1]),
        (zero_padding[2],zero_padding[2])
    ]
    minitf = tf.identity(np.reshape(mini,(R*C,)+mini.shape[2:]).astype(np.float32))
    minitf=tf.pad(minitf,ZERO_PAD)
    codetf = tf.identity(np.reshape(codebook,(R*C,-1)))
    codetf=codetf/tf.math.sqrt(tf.reduce_sum(codetf**2,axis=0,keepdims=True))
    ssq=tf.reduce_sum(minitf**2)

    # set up tf variables we need
    t=tf.Variable(np.zeros((R*C,3),dtype=np.float32))

    # this is how much we'll pad inside the translation functions
    zero_padding=np.require(zero_padding,dtype=np.int32)
    pad=tf.convert_to_tensor(zero_padding)

    # compute learning rate so that the first step the
    # GD algorithm moves t by at most "initial_step" in voxelspace
    l,g=calc_loss_and_grad(minitf,codetf,t,pad,interpolation_method=interpolation_method) 
    lr=initial_step/np.max(np.abs(g.numpy()))

    # run optimization
    opt=tf.optimizers.SGD(lr=lr,momentum=momentum)
    t_storage=[]
    losses=[]
    for i in misc.maybe_trange(niter,use_tqdm_notebook):
        t_storage.append(t.numpy()) # store value
        l,g=calc_loss_and_grad(minitf,codetf,t,pad,interpolation_method=interpolation_method) # calc loss
        losses.append(ssq+l.numpy()) # store loss
        opt.apply_gradients([(g,t)]) # make a step
        check_for_padding_issues(t,pad) # possibly raise an exception if t has gotten too big
        
    t_storage.append(t.numpy())
    l,g=calc_loss_and_grad(minitf,codetf,t,pad,interpolation_method=interpolation_method)
    losses.append(ssq+l.numpy())

    return t_storage,losses 


def check_for_padding_issues(t,pad):
    # help user diagnose padding issues
    tfl = tf.math.floor(t)
    tfl_min = tf.reduce_min(tfl,axis=0)
    tfl_max = tf.reduce_max(tfl,axis=0)
    lower_failure = (-pad[None,:]>tfl_min).numpy()
    upper_failure = (tfl_max >= pad[None,:]).numpy()
    failure = lower_failure | upper_failure
    if failure.any():
        rbad,cbad,mbad=np.array(np.where(failure)).T[0]
        v=tfl[rbad,cbad,mbad].numpy()
        thispad=pad[mbad].numpy()
        raise ValueError(f"t[{rbad},{cbad},{mbad}]={v} fails to satisfy -{thispad}<=<t<{thispad} -- try increasing zero_padding[{mbad}]")
    