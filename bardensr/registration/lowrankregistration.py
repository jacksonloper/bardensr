import numpy as np
import tensorflow as tf
from .. import misc
from . import translations_tf
from . import lowrankregistration_tf
import numbers
import scipy as sp
import scipy.optimize

def _procminicode(mini,codebook,zero_padding):
    minitf = tf.identity(mini)
    codetf = tf.identity(codebook)
    codetf = tf.cast(codebook,dtype=tf.float64)
    codetf = codetf/tf.math.sqrt(tf.reduce_sum(codetf**2,axis=0,keepdims=True))
    codetf = tf.cast(codetf,dtype=minitf.dtype)

    F=mini.shape[0]
    nd=len(mini.shape)-1

    if isinstance(zero_padding,numbers.Integral):
        zero_padding=(zero_padding,)*nd
    else:
        assert len(zero_padding)==nd
    zero_padding=np.require(zero_padding,dtype=np.int32)

    # this is how much we'll pad inside the translation functions
    sz=np.require(mini.shape[1:],dtype=np.int32)+zero_padding*2
    sz=tf.identity(sz)

    return minitf,codetf,zero_padding,sz

def calc_loss_and_grad(mini,codebook,t,zero_padding=10):
    '''
    numpy interface to the loss function optimized by
    lowrankregister.  this function sill be slow,
    because it performs quite a few preprocessing steps to convert
    from numpy-land to superfast tensorflow land.
    '''
    # get mini,code,zp ready for tensorflow
    minitf,codetf,zero_padding,sz=_procminicode(mini,codebook,zero_padding)

    # set up tf variables we need
    xt=tf.cast(tf.identity(t),dtype=tf.float64)

    # get loss
    l,g=lowrankregistration_tf._calc_loss_and_grad(minitf,codetf,xt,sz)

    # send back to numpyland
    return l.numpy(),g.numpy()


def precompile_tensorflow_code_for(mini,codebook,zero_padding=10):
    F=mini.shape[0]
    nd=len(mini.shape)-1
    t=np.zeros((F,nd))-zero_padding
    t=tf.identity(t)
    t=tf.cast(t,dtype=tf.float64)
    calc_loss_and_grad(mini,codebook,t,zero_padding)

def calc_loss(mini,codebook,t,zero_padding=10):
    '''
    numpy interface to the loss function optimized by
    lowrankregister.  this function sill be slow,
    because it performs quite a few preprocessing steps to convert
    from numpy-land to superfast tensorflow land.
    '''
    # get mini,code,zp ready for tensorflow
    minitf,codetf,zero_padding,sz=_procminicode(mini,codebook,zero_padding)

    # set up tf variables we need
    xt=tf.cast(tf.identity(t),dtype=tf.float64)

    # get loss
    l=lowrankregistration_tf._calc_scaled_loss(minitf,codetf,xt,sz)

    # send back to numpyland
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

        assert not np.isnan(lr),f'initial gradient magnitude={mag}'

        opt=tf.optimizers.SGD(momentum=momentum,learning_rate=lr)
        for i in range(maxiter):
            assert not np.isnan(t.numpy()).any()
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
            optimization_method='sgd',initial_guess=None,
            optimization_settings=None,
            compiled_functions=None):
    ts,losses,optim=_lowrankregister(mini,codebook,zero_padding=zero_padding,
            use_tqdm_notebook=use_tqdm_notebook,niter=niter,
            optimization_method=optimization_method,initial_guess=initial_guess,
            optimization_settings=optimization_settings,
            compiled_functions=compiled_functions)

    return ts[-1]

def _lowrankregister(mini,codebook,zero_padding=10,
            use_tqdm_notebook=False,niter=50,
            optimization_method='sgd',initial_guess=None,
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
        clag=lowrankregistration_tf._calc_loss_and_grad
    else:
        clag=compiled_functions.calc_loss_and_grad

    if optimization_settings is None:
        optimization_settings={}

    # get mini,code,zp ready for tensorflow
    minitf,codetf,zero_padding,sz=_procminicode(mini,codebook,zero_padding)

    # set up tf variables we need
    t=np.zeros((F,nd))-zero_padding  # e.g. grab floatingslice on interval [-10,shp+10]

    # if supplied, add in initial guess (with mean taken out)
    if initial_guess is not None:
        initial_guess=np.require(initial_guess,dtype=float)
        if initial_guess.shape!=(F,nd):
            raise ValueError(f"initial guess has wrong shape, should be {F} x {nd}")
        t=t+initial_guess-np.mean(initial_guess,axis=0,keepdims=True)

    t=tf.identity(t)
    t=tf.cast(t,dtype=tf.float64)

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
