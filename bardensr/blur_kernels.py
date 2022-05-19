import tensorflow as tf
import numpy as np

'''
 _     _
| |__ | |_   _ _ __
| '_ \| | | | | '__|
| |_) | | |_| | |
|_.__/|_|\__,_|_|

'''


def gaussian_filter_3d(X,sigmas):
    '''
    X -- ... x M0 x M1 x M2
    sigma -- tuple of length 3
    '''

    nd=len(X.shape)
    X=gaussian_filter_1d(X,sigmas[0],0)
    X=gaussian_filter_1d(X,sigmas[1],1)
    X=gaussian_filter_1d(X,sigmas[2],2)

    return X


def gaussian_filter_2d(X,sigmas):
    '''
    X -- ... x M0 x M1
    sigma -- tuple of length 2
    '''

    nd=len(X.shape)
    X=gaussian_filter_1d(X,sigmas[0],nd-2)
    X=gaussian_filter_1d(X,sigmas[1],nd-1)

    return X

def gaussian_filter_1d(X,sigma,axis):
    sigma=tf.cast(sigma,dtype=tf.float64)
    return tf.cond(
        sigma==tf.cast(0,tf.float64),
        lambda: X,
        lambda: _gaussian_filter_1d(X,sigma,axis),
    )

def _gaussian_filter_1d(X,sigma,axis):
    '''
    X -- tensor
    sigma -- scalar
    axis

    filters X over axis
    '''
    # construct filter (in float64 land)
    xs = tf.range(1,sigma*3+1,dtype=tf.float64)
    zero= tf.cast(0,dtype=tf.float64)[None]
    xs = tf.concat([-tf.reverse(xs,(0,)),zero,xs],axis=0)

    filt=tf.math.exp(-.5*xs**2/(sigma*sigma))
    filt=filt/tf.reduce_sum(filt)
    filt=filt[:,None,None] # width x 1 x 1

    # cast filter to X dtype
    filt=tf.cast(filt,dtype=X.dtype)

    # transpose X so that the spatial dimension is at the end
    axes=list(range(len(X.shape)))
    axes[-1],axes[axis]=axes[axis],axes[-1]
    X_transposed=tf.transpose(X,axes) # everythingelse x axis x 1

    # do convolution
    X_convolved_transposed=tf.nn.conv1d(X_transposed[None,...,None],filt,1,'SAME')[0,...,0]

    # transpose back
    X_convolved=tf.transpose(X_convolved_transposed,axes)

    return X_convolved

'''
     _
 ___| |__   __ _ _ __ _ __   ___ _ __
/ __| '_ \ / _` | '__| '_ \ / _ \ '_ \
\__ \ | | | (_| | |  | |_) |  __/ | | |
|___/_| |_|\__,_|_|  | .__/ \___|_| |_|
                     |_|
'''

def gaussian_sharpen_3d(X,sigmas,sharpening_levels):
    '''
    X -- ... x M0 x M1 x M2
    sigma -- tuple of length 3
    '''

    nd=len(X.shape)
    X=gaussian_sharpen_1d(X,sigmas[0],sharpening_levels[0],nd-3)
    X=gaussian_sharpen_1d(X,sigmas[1],sharpening_levels[1],nd-2)
    X=gaussian_sharpen_1d(X,sigmas[2],sharpening_levels[2],nd-1)

    return X


def gaussian_sharpen_2d(X,sigmas,sharpening_levels):
    '''
    X -- ... x M0 x M1
    sigma -- tuple of length 2
    '''

    nd=len(X.shape)
    X=gaussian_sharpen_1d(X,sigmas[0],sharpening_levels[0],nd-2)
    X=gaussian_sharpen_1d(X,sigmas[1],sharpening_levels[1],nd-1)

    return X

def gaussian_sharpen_1d(X,sigma,sharpening_level,axis):
    blurred= tf.cond(
        sigma==0,
        lambda: X,
        lambda: _gaussian_filter_1d(X,sigma,axis),
    )

    return X+ sharpening_level*(X-blurred)
