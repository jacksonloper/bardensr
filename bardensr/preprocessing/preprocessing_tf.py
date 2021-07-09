__all__=[
    'gaussian_filter_1d',
    'minmax',
    'decimate_1d',
    'grad_afd',
    'denoising_prepare_for_split_bregman',
    'denoising_split_bregman_update',
    'ising_model_meanfield_update'
]

import collections
import dataclasses
import typing


import tensorflow as tf
import tensorflow_probability as tfp

from bardensr import misc
from bardensr import tf_helpers

def decimate_1d(X,n,axis):
    shp=tf.shape(X)
    shp2=tf.concat([shp[:axis],[shp[axis]//n,n],shp[axis+1:]],0)
    X=tf.reshape(X,shp2)
    X=tf.reduce_mean(X,axis=axis+1)
    return X

def grad_afd(X,axes):
    '''
    estimates the gradient of X along axes for all interior voxels
    with an asymmetrical finite difference.

    Input is

    - (M0 x M1 x M2 ... x M(n-1))
    - axes, m unique integers in {0,1,...n-1}

    Output is (M0' x M1' x ... x M(n-1)' x m), where
    Mi' is shorter by 1 if it is referred to by axes.

    '''

    shp=tf.shape(X)
    n=tf.shape(shp)[0]
    axshapes=tf.gather(shp,axes)
    naxes=len(axes)


    axwrap=[[ax] for ax in axes]
    zero_naxes=tf.zeros(naxes,dtype=tf.int32)
    stv_base=tf.scatter_nd(axwrap,zero_naxes,(n,))

    szv_base=tf.cast(tf.fill((n,),-1),dtype=tf.int32)
    szv_base=tf.tensor_scatter_nd_update(szv_base,axwrap,axshapes-1)

    diffs=[]
    for i,axis in enumerate(axes):
        # first perform difference
        sl1,sz1=tf_helpers.construct_1dslice(shp,1,shp[axis]-1,axis)
        sl2,sz2=tf_helpers.construct_1dslice(shp,0,shp[axis]-1,axis)
        df=tf.slice(X,sl1,sz1)-tf.slice(X,sl2,sz2)

        # next, trim along all other axes
        stv=tf.tensor_scatter_nd_update(stv_base,[[axis]],[0])
        szv=tf.tensor_scatter_nd_update(szv_base,[[axis]],[-1])
        df=tf.slice(df,stv,szv)
        diffs.append(df)
    return tf.stack(diffs,axis=-1)

def sum_neighbors(x,axes):
    '''
    at each index, compute the sum of values in x which are adjacent to that index
    '''
    shp=tf.shape(x)
    n=tf.shape(shp)[0]

    ys=[]
    for i in axes:
        LEFT = tf_helpers.construct_1dslice(shp,1,1,i)
        MID1 =  tf_helpers.construct_1dslice(shp,0,shp[i]-2,i)
        MID2 =  tf_helpers.construct_1dslice(shp,2,shp[i]-2,i)
        RIGHT = tf_helpers.construct_1dslice(shp,shp[i]-2,1,i)
        ys.append(tf.concat([
            tf.slice(x,*LEFT),
            tf.slice(x,*MID1)+tf.slice(x,*MID2),
            tf.slice(x,*RIGHT),
        ],axis=i))
    return tf.math.accumulate_n(ys)

class SplitBregmanDenoiserState(typing.NamedTuple):
    target:tf.Tensor
    u:tf.Tensor
    d:tf.Tensor
    b:tf.Tensor
    gamma:tf.Tensor
    rho:tf.Tensor
    loss:tf.Tensor

class DenoisingLoss(typing.NamedTuple):
    f:tf.Tensor
    g:tf.Tensor
    loss:tf.Tensor

def evaluate_binned_function(x,edges,vals):
    '''
    Evaluates the piecewise constant function defined by (edges,vals)
    on x, i.e. something like y[i] = vals[bin which x[i] falls into]

    Input

    - x, a tensor
    - bins, a 1d tensor
    - vals, a 1d tensor with len(vals)=len(bins)-1

    Output is y, same shape as x, taking values from vals.
    '''

    # flatten
    flat=tf.reshape(x,(tf.reduce_prod(tf.shape(x)),))

    # decide which bin to put each x in
    bin_idxs=tf.searchsorted(edges,flat)

    # gather
    gth = tf.gather(vals,bin_idxs)

    # reshape and done
    return tf.reshape(gth,x.shape)

class HistogramBinaryClassifier(typing.NamedTuple):
    edges: tf.Tensor
    llrs: tf.Tensor
    logp_plus: tf.Tensor
    logp_minus: tf.Tensor

def estimate_histogram_binary_classifier(vals,n_edges,logit_weights,pseudocount):
    '''
    Consider the conditional model family for (X|Z) defined by::

        X|Z=1  ~ p+
        X|Z=-1 ~ p-

    This function uses samples of X to build an approximate neyman-pearson classifier for
    estimating Z|X, using pseudocounted histogram estimates for p+ and p-.  The classifier
    is returned in the form of a collection of bins (indicated by edges) and
    a collection of logits (indicating log p+(x)/p-(x) for values of x inside
    each bin).

    Input

    - vals, tensor
    - n_edges, integer
    - logit_weights, indicating ground truth for log p(Z=1|X)/log p(Z=-1|X).

    Output is a HistogramBinaryClassifier object.
    '''

    MN=tf.reduce_min(vals)
    MX=tf.reduce_max(vals)
    edges=tf.range(MN,MX,(MX-MN)/n_edges)

    p=tf.math.sigmoid(logit_weights)
    counts_plus=tf.math.log(tfp.stats.histogram(vals,edges,weights=p) + pseudocount)
    counts_minus=tf.math.log(tfp.stats.histogram(vals,edges,weights=1-p)+pseudocount)

    pmf_plus = counts_plus - tf.math.reduce_logsumexp(counts_plus)
    pmf_minus = counts_minus - tf.math.reduce_logsumexp(counts_minus)

    return HistogramBinaryClassifier(edges,pmf_plus - pmf_minus,pmf_plus,pmf_minus)


def ising_model_meanfield_update(beta,op,zeta,mask):
    '''
    Consider the PMF on x in {-1,1}^n with

        log p(x) = c + tf.reduce_sum(x*op(x)) + tf.reduce_sum(beta*x)

    Consider the model family

        log q(x;zeta) = tf.reduce_sum(x*zeta - tf.math.log(2*tf.math.cosh(zeta)))

    Consider the problem of minmizing KL(q||p).

    This function performs the update

        newzeta = (1-mask)*zeta + mask*(beta+tanh(op(zeta)))

    If mask is sufficiently close to zero or represents a graph coloring
    that respects the sparsity pattern of op, then this update will
    make progress on the target objective.

    Input

    - beta, a tensor
    - op, a symmetric linear operator, accepting tensors like beta
    - zeta, a tensor of the same shape as beta
    - mask, a tensor of the same shape of beta, with values in the interval [0,1]

    Output is newzeta, a new value of zeta.  Under favorable conditions,
    KL(q(;newzeta)||p) <= KL(q(;zeta)||p)

    '''

    return (1-mask)*zeta + mask*(beta+tf.math.tanh(op(zeta)))

def denoising_loss(target,u,op,rho):
    '''
    Evaluates::

        out = f(u) + rho * g(u)
        f(u) = .5 *tf.reduce_sum((u-target)**2)
        g(u) = tf.reduce_sum(tf.sqrt(tf.reduce_sum(op(u)**2,axis=-1)))

    Output is a DenoisingLoss object containing f, g, and f+rho * g
    '''

    f = .5 *tf.reduce_sum((u-target)**2)
    g = tf.reduce_sum(tf.sqrt(tf.reduce_sum(op(u)**2,axis=-1)))

    return DenoisingLoss(f,g,f+rho*g)


def denoising_prepare_for_split_bregman(target,op,rho,gamma):
    '''
    Initialize state for using a split bregman method to minimize
    denoising_loss(target,op,rho) with learning rate gamma.

    Input:

    - target (want to find u which is close to target)
    - op     (want op(target) to be small)
    - rho    (strength with which you want op(target) to be small)
    - gamma  (learning rate of split bregman method, 5.0 usually works ok)

    Output is a SplitBregmanDenoiserState object which can be fed into
    denoising_split_bregman_update.
    '''

    u=target
    d=op(target)*0
    b=d
    loss=denoising_loss(target,u,op,rho).loss

    return SplitBregmanDenoiserState(
        target,
        u,
        d,
        b,
        tf.cast(gamma,dtype=target.dtype),
        tf.cast(rho,dtype=target.dtype),
        loss,
    )

def cone_shrink(a,gamma):
    '''
    solve

      argmin_d  sum(sqrt(sum(d**2,axis=-1))) + .5*gamma * ||d-a||^2
    '''

    mags=tf.math.sqrt(tf.reduce_sum(a**2,axis=-1))
    shrinks=tf.math.maximum(tf.cast(0,dtype=mags.dtype),1-(1/(gamma*mags)))

    return a*shrinks[...,None]

def denoising_split_bregman_update(state,op):
    '''
    perform a split bregman update towards minimizing
    denoising_loss(state.target,op,state.rho).  That is:

        u <- min .5*||u-target||^2 + .5*gamma*||op(u)-(d-b)||^2
        d <- sum(sqrt(sum(d**2,axis=-1))) + 5*gamma*||op(u)-(d-b)||^2
        b <- b - d + op(u)

    Input

    - state, a SplitBregmanDenoiserState object (e.g. created by denoising_prepare_for_split_bregman)
    - op, the regularization operator

    Output is a new SplitBregmanDenoiserState object
    storing the new values of u,d,b after the split bregman update.

    '''
    # u update -- take a step towards minimizing .5*||u-target||^2 + .5*gamma*||op(u)-(d-b)||^2
    def augmented_u_loss(u,d,b):
        loss = .5*tf.reduce_sum((u-state.target)**2)
        loss = loss + .5*state.gamma*tf.reduce_sum((op(u) - (d-b))**2)
        return loss
    with tf.GradientTape() as t:
        t.watch(state.u)
        loss=augmented_u_loss(state.u,state.d,state.b)
    search_direction=t.gradient(loss,state.u)
    sd_norm = tf.reduce_sum(search_direction**2)
    sd_op_norm = tf.reduce_sum(op(search_direction)**2)
    travelling_distance = sd_norm / (sd_norm + state.gamma*sd_op_norm)
    u=state.u-travelling_distance*search_direction

    # d update -- optimize sum(sqrt(sum(d**2,axis=-1))) + 5*gamma*||op(u)-(d-b)||^2
    d = cone_shrink(op(u)+state.b,state.gamma/state.rho)

    # lagrange multiplier update
    b=state.b-d+op(u)

    # get new loss
    loss=denoising_loss(state.target,u,op,state.rho).loss

    return SplitBregmanDenoiserState(state.target,u,d,b,state.gamma,state.rho,loss)

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
        bl=gaussian_filter_1d(X, s, ax)

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
