import tensorflow as tf

from .. import blur_kernels

@tf.function
def mnmx(X,axes):
    '''
    Input
    - X -- M0 x M1 x M2 x ... x M(n-1)
    - axes -- set of integers in {0,1,...n-1}

    normalize by min and max along axes
    '''

    X=X-tf.reduce_min(X,axis=axes,keepdims=True)
    X=X/tf.reduce_max(X,axis=axes,keepdims=True)

    return X

@tf.function
def mnmx_background_subtraction(X,axes,sigmas):
    '''
    Input4
    - X -- M0 x M1 x M2 ... x M(n-1)
    - axes -- set of integers in {0,1,...n-1}
    - blurs -- corresponding floating points

    this
    1. runs gaussian background subtraction along axes sigmas
    2. normalizes by min and max along axes
    '''

    X=mnmx(X,axes)

    bl=X
    for s,ax in zip(sigmas,axes):
        bl=blur_kernels.gaussian_filter_1d(X, s, ax)

    X=X-bl
    X=tf.clip_by_value(X,0,X.dtype.max)

    X=mnmx(X,axes)

    return X
