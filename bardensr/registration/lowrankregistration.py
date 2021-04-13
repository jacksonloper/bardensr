import numpy as np
import tensorflow as tf
from .. import misc
from .. import kernels
import numbers
import scipy as sp
import scipy.optimize

def _calc_loss(X,code,t,sz):
    '''
    Proportion of the variance unexplained by using
    a single code to explain each pixel.
    '''
    newX = kernels.floating_slices(X,t,sz,'hermite')
    dots=tf.einsum('fj,f...->...j',code,newX)

    mx = tf.reduce_max(dots,axis=-1)
    # weights = tf.nn.softmax(dots,axis=-1)
    # mx = dots*weights

    loss=1-tf.reduce_sum(mx**2)/tf.reduce_sum(X**2)
    return loss

def _calc_loss_and_grad(X,code,t,sz):
    '''
    X -- F x M0 x M1 x ... Mn
    code -- F x J
    t -- F x n
    sz -- n
    '''

    with tf.GradientTape() as tape:
        tape.watch(t)
        loss=_calc_loss(X,code,t,sz)

    grad=tape.gradient(loss,t)
    return loss,grad


def compile_lowrank_registration_functions(RC,nd=3):
    class LowrankRegistrationFunctions(tf.Module):
        @tf.function
        def calc_loss_and_grad(self,X,code,t,sz):
            return _calc_loss_and_grad(X,code,t,sz)

    cw=LowrankRegistrationFunctions()
    cw.calc_loss_and_grad.get_concrete_function(
        tf.TensorSpec([RC,None,None,None]),
        tf.TensorSpec([RC,None]),
        tf.TensorSpec([RC,3]),
        tf.TensorSpec([3],dtype=tf.int32),
    )
    return cw

def calc_loss_and_grad(mini,codebook,t,zero_padding=10):
    '''
    numpy interface to the loss function optimized by
    lowrankregister.  this function sill be slow,
    because it performs quite a few preprocessing steps to convert
    from numpy-land to superfast tensorflow land.
    '''
    F=mini.shape[0]
    nd=len(mini.shape)-1

    if isinstance(zero_padding,numbers.Integral):
        zero_padding=(zero_padding,)*nd
    else:
        assert len(zero_padding)==nd

    minitf = tf.identity(mini.astype(np.float32))
    codetf = tf.identity(codebook.astype(np.float32))
    codetf=codetf/tf.math.sqrt(tf.reduce_sum(codetf**2,axis=0,keepdims=True))

    # set up tf variables we need
    zero_padding=np.require(zero_padding,dtype=np.int32)
    xt=tf.convert_to_tensor(t,dtype=tf.float32)

    # this is how much we'll pad inside the translation functions
    sz=np.require(mini.shape[1:],dtype=np.int32)+zero_padding*2
    sz=tf.convert_to_tensor(sz,dtype=tf.int32)

    l,g=_calc_loss_and_grad(minitf,codetf,xt,
        sz,interpolation_method=interpolation_method)
    return l.numpy(),g.numpy()


def calc_loss(mini,codebook,t,zero_padding=10):
    '''
    numpy interface to the loss function optimized by
    lowrankregister.  this function sill be slow,
    because it performs quite a few preprocessing steps to convert
    from numpy-land to superfast tensorflow land.
    '''
    F=mini.shape[0]
    nd=len(mini.shape)-1

    if isinstance(zero_padding,numbers.Integral):
        zero_padding=(zero_padding,)*nd
    else:
        assert len(zero_padding)==nd

    minitf = tf.identity(mini.astype(np.float32))
    codetf = tf.identity(codebook.astype(np.float32))
    codetf=codetf/tf.math.sqrt(tf.reduce_sum(codetf**2,axis=0,keepdims=True))

    # set up tf variables we need
    zero_padding=np.require(zero_padding,dtype=np.int32)
    xt=tf.convert_to_tensor(t,dtype=tf.float32)

    # this is how much we'll pad inside the translation functions
    sz=np.require(mini.shape[1:],dtype=np.int32)+zero_padding*2
    sz=tf.convert_to_tensor(sz,dtype=tf.int32)

    l=_calc_loss(minitf,codetf,xt,sz)
    return l.numpy()


def minimize(method,inner_func,t,maxiter,momentum=0.8,lr=None,first_step_size=.1,first_step_size_norm='max',counter=None):
    losses=[]
    ts=[]
    tru_shape=t.shape

    if method=='bfgs':
        def func(x):
            xt=tf.convert_to_tensor(x.reshape(tru_shape))
            l,g=inner_func(xt)
            losses.append(l.numpy())
            ts.append(xt.numpy())
            if counter is not None:
                counter.set_description(f'variance unexplained={np.min(losses):.4f}')
            return l.numpy(),g.numpy().ravel()

        def cb(*args,**kwargs):
            if counter is not None:
                counter.update(1)

        return ts,losses,sp.optimize.minimize(func,t,jac=True,
                options=dict(maxiter=maxiter),callback=cb,method='bfgs')
    elif method=='sgd':
        t=tf.Variable(t)

        if lr is None:
            # pick LR by getting first step size
            l,g=inner_func(tf.convert_to_tensor(t))
            if first_step_size_norm=='max':
                mag=np.abs(g.numpy()).max()
            elif first_step_size_norm=='l2':
                mag=np.sqrt(np.sum(g.numpy()**2))
            else:
                raise NotImplementedError(first_step_size_norm)
            lr=first_step_size/mag

        opt=tf.optimizers.SGD(momentum=momentum,lr=lr)
        for i in range(maxiter):
            l,g=inner_func(tf.convert_to_tensor(t))
            losses.append(l.numpy())
            ts.append(t.numpy().copy())
            if counter is not None:
                counter.update(1)
                counter.set_description(f'variance unexplained={np.min(losses):.4f}')
            opt.apply_gradients([(g,t)])
        return ts,losses,None
    else:
        raise NotImplementedError(method)


def lowrankregister(mini,codebook,zero_padding=10,
            use_tqdm_notebook=False,niter=50,
            optimization_method='sgd',
            optimization_settings=None,
            compiled_functions=None):
    ts,losses,optim=_lowrankregister(mini,codebook,zero_padding=zero_padding,
            use_tqdm_notebook=use_tqdm_notebook,niter=niter,
            optimization_method=optimization_method,
            optimization_settings=optimization_settings,
            compiled_functions=compiled_functions)
    return ts[-1]

def _lowrankregister(mini,codebook,zero_padding=10,
            use_tqdm_notebook=False,niter=50,
            optimization_method='sgd',
            optimization_settings=None,
            compiled_functions=None):
    '''
    Input
    * mini          -- F x M0 x M1 x M2 x ... M(n-1)
    * codebook      -- F x J
    * zero_padding  -- n or scalar

    Output
    * ts -- niter x F x 3
    * losses -- niter

    '''
    F=mini.shape[0]
    nd=len(mini.shape)-1

    if compiled_functions is None:
        clag=_calc_loss_and_grad
    else:
        clag=compiled_functions.calc_loss_and_grad

    if optimization_settings is None:
        optimization_settings={}

    if isinstance(zero_padding,numbers.Integral):
        zero_padding=(zero_padding,)*nd
    else:
        assert len(zero_padding)==nd

    if tf.is_tensor(mini):
        minitf = tf.cast(tf.identity(mini),dtype=tf.float32)
    else:
        minitf = tf.identity(mini.astype(np.float32))

    if tf.is_tensor(codebook):
        codetf = tf.identity(codebook.astype(np.float32))
    else:
        codetf = tf.cast(tf.identity(codebook),dtype=tf.float32)

    codetf=codetf/tf.math.sqrt(tf.reduce_sum(codetf**2,axis=0,keepdims=True))

    # set up tf variables we need
    zero_padding=np.require(zero_padding,dtype=np.int32)
    t=np.zeros((F,nd))-zero_padding
    t=tf.convert_to_tensor(t,dtype=tf.float32)

    # this is how much we'll pad inside the translation functions
    sz=np.require(mini.shape[1:],dtype=np.int32)+zero_padding*2
    sz=tf.convert_to_tensor(sz,dtype=tf.int32)


    # gradient descent
    def inner_func(xt):
        return clag(minitf,codetf,xt,sz)

    if use_tqdm_notebook:
        import tqdm.notebook
        with tqdm.notebook.tqdm(total=niter) as counter:
            ts,losses,minresult=minimize(
                optimization_method,
                inner_func,t,niter,counter=counter,
                **optimization_settings)
    else:
        ts,losses,minresult=minimize(
                optimization_method,
                inner_func,t,niter,
                **optimization_settings)

    best=np.argmin(losses)

    ts.append(ts[best])
    losses.append(losses[best])

    return ts,losses,minresult

    # # compute learning rate so that the first step the
    # # GD algorithm moves t by at most "initial_step" in voxelspace
    # l,g=calc_loss_and_grad(minitf,codetf,t,sz,interpolation_method=interpolation_method)
    # lr=initial_step/np.max(np.abs(g.numpy()))

    # # run optimization
    # opt=tf.optimizers.SGD(lr=lr*.1,momentum=momentum)
    # t_storage=[]
    # losses=[]
    # for i in misc.maybe_trange(niter,use_tqdm_notebook):
    #     t_storage.append(t.numpy()) # store value
    #     l,g=calc_loss_and_grad(minitf,codetf,t,sz,interpolation_method=interpolation_method) # calc loss
    #     losses.append(ssq+l.numpy()) # store loss
    #     opt.apply_gradients([(g,t)]) # make a step

    # t_storage.append(t.numpy())
    # l,g=calc_loss_and_grad(minitf,codetf,t,sz,interpolation_method=interpolation_method)
    # losses.append(ssq+l.numpy())

    # return t_storage,losses
