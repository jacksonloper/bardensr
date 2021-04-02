from . import kernels
import tensorflow as tf
from . import misc
import scipy as sp
import scipy.ndimage
import numpy as np

def calc_valid_region(shp,t,interpolation_method='hermite'):
    '''
    Input
    * shp -- n-vector (int)
    * t   -- kxn (floating)

    Output
    * newt -- kxn-vector floating
    * sz -- n-vector (int)

    Let X be a tensor of shape

        k x M0 x M1 x ... Mn

    Find vector "adjustment" and size "sz" so that we can grab slices
    of size sz starting from newt=t-adjustment...

        X[k,t[k,0]-adjustment[0]:t[k,0]-adjustment[0]+sz[0]]

    from each k, using interpolation_method for non-integer values of t,
    without looking OOB for X.  Indeed, we find the most negative value
    of adjustment and the most positive value of sz so that this can
    be done.
    '''

    shp=tf.convert_to_tensor(shp)
    t=tf.convert_to_tensor(t)

    if interpolation_method=='hermite':
        ld=1
        rd=1
    elif interpolation_method=='linear':
        ld=0
        rd=0
    else:
        raise NotImplementedError()

    # need t[k,0]-adjustment[0]>=ld
    # adjustment <= t[k,0] - ld
    adjustment = tf.reduce_min(t-ld,axis=0) # <-- this is floatingpoint

    # need t[k,0] - adjustment[0] + sz[0] < shape[0] - rd
    # sz[0] < shape[0] - rd + adjustment[0] - t[k,0]
    amt = tf.reduce_min(adjustment[None,:]-t,axis=0)
    sz = shp-rd + tf.cast(tf.math.ceil(amt)-1,dtype=shp.dtype)

    return t-adjustment[None,:],sz

def calc_complete_region(shp,t,interpolation_method='hermite'):
    '''
    Input
    * shp -- n-vector (int)
    * t   -- kxn (floating)

    Output
    * newt -- kxn-vector floating
    * sz -- n-vector (int)

    Let X be a tensor of shape

        k x M0 x M1 x ... Mn

    Find vector "adjustment" and size "sz" so that we can grab slices
    of size sz starting from newt=t-adjustment...

        X[k,t[k,0]-adjustment[0]:t[k,0]-adjustment[0]+sz[0]]

    from each k, using interpolation_method for non-integer values of t,
    in such a way that every value of X which can be properly interpolated.
    '''

    shp=tf.convert_to_tensor(shp)
    t=tf.convert_to_tensor(t)

    # need t[k,0]-adjustment[0]<=0
    # adjustment >= t[k,0]
    adjustment = tf.reduce_max(t,axis=0) # <-- this is floatingpoint

    # need t[k,0] - adjustment[0] + sz[0] >= shape[0]
    # sz >= shape[0] + adjustment[0] - t[k,0]
    amt = tf.reduce_max(adjustment[None,:]-t,axis=0)
    sz = shp + tf.cast(tf.math.ceil(amt),dtype=shp.dtype)

    return t-adjustment[None,:],sz


def calc_loss_and_grad(X,code,t,sz,interpolation_method='hermite'):
    '''
    X -- F x M0 x M1 x M2
    code -- F x J
    t -- F x 3
    sz -- 3
    '''

    with tf.GradientTape() as tape:
        tape.watch(t)
        newX = kernels.floating_slices(X,t,sz,interpolation_method)

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


def apply_registration(mini,totalt,mode='valid',interpolation_method='linear'):
    R,C=mini.shape[:2]
    if mode=='valid':
        newt,sz=calc_valid_region(mini.shape[2:],
            totalt,interpolation_method)
    elif mode=='complete':
        newt,sz=calc_complete_region(mini.shape[2:],
            totalt,interpolation_method)
    else:
        raise NotImplementedError()
    minir=kernels.floating_slices(mini.reshape((R*C,)+mini.shape[2:]),
            newt,sz,interpolation_method)
    minir=minir.numpy().reshape((R,C)+minir.shape[1:])

    return minir

def register(mini,codebook,zero_padding=(0,0,0),
                    use_tqdm_notebook=False,niter=100,initial_step=.1,momentum=.9,
                    interpolation_method='hermite'):
    R,C=mini.shape[:2]

    minitf = tf.identity(np.reshape(mini,(R*C,)+mini.shape[2:]).astype(np.float32))
    codetf = tf.identity(np.reshape(codebook,(R*C,-1)).astype(np.float32))
    codetf=codetf/tf.math.sqrt(tf.reduce_sum(codetf**2,axis=0,keepdims=True))
    ssq=tf.reduce_sum(minitf**2)

    # set up tf variables we need
    t=tf.Variable(np.zeros((R*C,3),dtype=np.float32))

    # this is how much we'll pad inside the translation functions
    sz=np.require(mini.shape[2:],dtype=np.int32)+np.require(zero_padding)
    sz=tf.convert_to_tensor(sz,dtype=tf.int32)

    # compute learning rate so that the first step the
    # GD algorithm moves t by at most "initial_step" in voxelspace
    l,g=calc_loss_and_grad(minitf,codetf,t,sz,interpolation_method=interpolation_method)
    lr=initial_step/np.max(np.abs(g.numpy()))

    # run optimization
    opt=tf.optimizers.SGD(lr=lr*.1,momentum=momentum)
    t_storage=[]
    losses=[]
    for i in misc.maybe_trange(niter,use_tqdm_notebook):
        t_storage.append(t.numpy()) # store value
        l,g=calc_loss_and_grad(minitf,codetf,t,sz,interpolation_method=interpolation_method) # calc loss
        losses.append(ssq+l.numpy()) # store loss
        opt.apply_gradients([(g,t)]) # make a step

    t_storage.append(t.numpy())
    l,g=calc_loss_and_grad(minitf,codetf,t,sz,interpolation_method=interpolation_method)
    losses.append(ssq+l.numpy())

    return t_storage,losses
