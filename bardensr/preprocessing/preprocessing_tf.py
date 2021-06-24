__all__=[
    'gaussian_filter_1d',
    'minmax',
]

import collections
import tensorflow as tf

from .. import blur_kernels
from .. import misc

def gaussian_filter_1d(X,sigma,axis):
    '''
    Gaussian filter along a single axis of a tensor

    Input:

    - X (a tensor with a float-ish dtype)
    - sigma (will be cast to float64)
    - axis (integer)

    Output: tensor with the same shape and dtype as "X",
    blurred along "axis" with a gaussian filter with
    "sigma" pixels of standard deviation.
    '''
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


LucyRichardsonResult =collections.namedtuple('LucyRichardsonResult', ['X','losses'])

def lucy_richardson(target,guess,op,opT=None,use_tffunc=True,
                    niter=100,thresh=1e-10,
                    use_tqdm_notebook=False,
                    record_losses=False
                ):

    # if opT is None, we assume op is symmetric
    if opT is None:
        opT=op

    # need op_onseys for lucy richardson!
    ones = tf.ones_like(target)
    opT_oneseys = opT(ones)

    # get zero and thresh in correct dtypes
    zero=tf.cast(0,target.dtype)
    thresh=tf.cast(thresh,target.dtype)

    # setup function decorators (maybe use tffunc)
    tffunc = tf.function if use_tffunc else (lambda x: x)

    # setup losses (None if we don't record, list otherwise)
    losses = [] if record_losses else None

    # setup loss and iteration functions
    @tffunc
    def calc_loss(x_guess):
        '''
        - log Poisson(target ; op(X))
        '''
        blx=op(x_guess)
        logblxtarget = tf.where(target<thresh,zero,tf.math.log(blx)*target)
        logp = blx - logblxtarget

        # unfortunately we have to cast to float32 at least
        # or there is no hope whatsoever of this being accurate
        return tf.reduce_mean(tf.cast(logp,dtype=tf.float32))

    @tffunc
    def lucy_richardson(x_guess):
        '''
        basically

          x = x * op(target/op(x)) / op_oneseys

        except with a little numerical safeguard in case op(x) gets too close to zero
        '''
        blurx=op(x_guess)
        return x_guess * opT(tf.where(blurx<thresh,zero,target/blurx)) / opT_oneseys

    # run iterations
    for i in misc.maybe_trange(niter,use_tqdm_notebook):
        guess=lucy_richardson(guess)
        if record_losses:
            losses.append(calc_loss(guess).numpy())

    return LucyRichardsonResult(guess,losses)

@tf.function
def minmax(X,axes):
    '''
    Subtract minimum and divide by maximum
    along axes.

    Input:

    - X (M0 x M1 x M2 x M3 ... x M(n-1))
    - axes (integers in the set {0,1,...{n-1})

    Output: tensor with the same shape and dtype as X,
    except normalized along axes.  In brief::

        X=X-tf.reduce_min(X,axis=axes,keepdims=True)
        X=X/tf.reduce_max(X,axis=axes,keepdims=True)
        return X

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

def background_subtraction(X,axes,sigmas):
    '''
    Input4
    - X -- M0 x M1 x M2 ... x M(n-1)
    - axes -- set of integers in {0,1,...n-1}
    - blurs -- corresponding floating points

    this
    1. runs gaussian background subtraction along axes sigmas
    2. normalizes by min and max along axes
    '''

    bl=X
    for s,ax in zip(sigmas,axes):
        bl=gaussian_filter_1d(X, s, ax)

    X=X-bl
    X=tf.clip_by_value(X,0,X.dtype.max)

    return X
