import tensorflow as tf
import numpy as np

def heat_kernel(X,niter,axis=0):
    pad_width=[(0,0) for i in range(len(X.shape))]
    pad_width[axis]=(niter,niter)
    pad_width=tf.convert_to_tensor(pad_width)
    X=tf.pad(X,pad_width)
    sl=[slice(0,None) for i in range(len(X.shape))]
    sl1=list(sl); sl1[axis]=slice(1,None)
    sl2=list(sl); sl2[axis]=slice(None,-1)
    for i in range(niter*2):
        X=.5*(X[sl1]+X[sl2])
    return X


@tf.function(autograph=False)
def heat_kernel_nd(X,niters):
    for i in range(len(niters)):
        if niters[i]>0:
            X=heat_kernel(X,niters[i],axis=i)
    return X


@tf.function(autograph=False)
def gaussian_filter_3d(X,sigmas):
    '''
    X -- ... x M0 x M1 x M2
    sigma -- tuple of length 3
    '''

    nd=len(X.shape)
    X=gaussian_filter_1d(X,sigmas[0],nd-3)
    X=gaussian_filter_1d(X,sigmas[1],nd-2)
    X=gaussian_filter_1d(X,sigmas[2],nd-1)

    return X

def gaussian_filter_1d(X,sigma,axis):
    '''
    X -- tensor
    sigma -- scalar
    axis

    filters X over axis
    '''
    xs=tf.cast(tf.range(-sigma*3+1,sigma*3+2),dtype=X.dtype)
    filt=tf.math.exp(-.5*xs**2/(sigma*sigma))
    filt=filt/tf.reduce_sum(filt)
    filt=filt[:,None,None] # width x 1 x 1

    # now we got to transpose X annoyingly

    axes=list(range(len(X.shape)))
    axes[-1],axes[axis]=axes[axis],axes[-1]

    X_transposed=tf.transpose(X,axes) # everythingelse x axis x 1

    newshp=(np.prod(X_transposed.shape[:-1]),X_transposed.shape[-1],1)
    X_transposed_reshaped=tf.reshape(X_transposed,newshp)

    X_convolved=tf.nn.conv1d(X_transposed_reshaped,filt,1,'SAME')
    X_convolved_reshaped=tf.reshape(X_convolved,X_transposed.shape)

    X_convolved_reshaped_transposed=tf.transpose(X_convolved_reshaped,axes)

    return X_convolved_reshaped_transposed
