__all__=[
    'find_translations_using_model',
    'apply_translations',
    'distributed_translation_estimator',
    'pairwise_correlation_registration',
    'dotproducts_at_translations'
]

import collections
import dataclasses
import numbers
import typing
from . import translations_tf
import tensorflow as tf
from .. import misc
from ..rectangles import tiling
import scipy as sp
import scipy.ndimage
import numpy as np
import scipy.signal

from .analytic_centering import calc_reasonable_rectangles
from .lowrankregistration import lowrankregister


@dataclasses.dataclass
class FindTranslationsUsingModelResults:
    '''
    Information about an attempt to translate frames
    together according to a low rank model.
    '''

    corrections: np.ndarray
    losses: np.ndarray


def find_translations_using_model(imagestack,codebook,maximum_wiggle=10,niter=50,
                                    use_tqdm_notebook=False,initial_guess=None):
    '''
    A method that uses the codebook and the model to find a
    translation of the imagestack which is more consistent with
    the observation model.  Before running this code, we generally
    advocate preprocessing by running `bardensr.preprocess_minmax`,
    running `bardensr.preprocess_bgsubtraction` and then running
    `bardensr.preprocess_minmax` again.

    Input

    - imagestack (N x M0 x M1 x M2 numpy array)
    - codebook (N x J numpy array)
    - [optional] maximum_wiggle (tuple of 3 integers;
      default (10,10,10); maximum possible wiggle
      permitted along each spatial dimension)
    - [optional] niter (integer; default 50; number of
      rounds of gradient descent to run in estimating
      the registration)

    Output: a FindTranslationsUsingModelResults, including

    - corrections (N x 3 numpy array, indicating how each imagestack
      should be shifted)
    - losses (loss computed at each iteration, indicating how well
      the gradient descent is proceeding)

    '''

    nframes=imagestack.shape[0]
    n=len(imagestack.shape)-1
    if imagestack.shape[0]==1:
        raise ValueError("this imagestack has only one frame; it is meaningless"
                        "to try to register one frame")


    # check for zero-indices
    nonzero_guys = tuple([i for i in range(1,n+1) if imagestack.shape[i]>1])
    nonzero_guys_m1 = tuple([(i-1) for i in range(1,n+1) if imagestack.shape[i]>1])
    zero_guys_m1 = tuple([(i-1) for i in range(1,n+1) if imagestack.shape[i]==1])

    if initial_guess is not None:
        if not np.allclose(initial_guess[:,zero_guys_m1],initial_guess[0,zero_guys_m1]):
            raise ValueError(
                "data has no extent along dimensions {str(zero_guys_m1)} "
                "but initial guess varies along some of those dimensions")
    else:
        initial_guess=np.zeros((nframes,n))

    imagestack_sq=np.squeeze(imagestack)
    ts,losses,optim=lowrankregister(imagestack_sq,codebook,
        zero_padding=maximum_wiggle,niter=niter,
        use_tqdm_notebook=use_tqdm_notebook,
        initial_guess=initial_guess[:,nonzero_guys_m1])

    corrections=ts[-1]
    corrections=corrections-np.mean(corrections,axis=0,keepdims=True)

    result=np.zeros((codebook.shape[0],n))
    for i,j in enumerate(nonzero_guys):
        result[:,j-1]=corrections[:,i]

    return FindTranslationsUsingModelResults(
        result,
        losses
    )


def apply_translations(imagestack,corrections,mode='valid',interpolation_method='linear'):
    '''
    Apply (potentially non-integer) translations to each frame in an imagestack.

    Input:

    - imagestack (N x M0 x M1 x M2 numpy array)
    - corrections (N x 3 floating point numpy)
    - mode ('valid' or 'full'; this indicates what to do with voxels
      for which not all frames have been measured.  valid trims them
      out, full sets them to zero.)
    - interpolation_method ('hermite' or 'linear' or 'nearest';
      how to deal with cases where corrections are not integers)

    Output:

    - imagestack2 (N x M0' x M1' x M2')
    - trimmed_corrections (N x 3 array, indicating the coordinates in imagestack
      which are used to supply imagestack2[:,0,0,0]).  This will be
      the same as corrections, up to an overall shift which may be applied
      to every row independently.  In particular,
      trimmed_corrections[i,j]-trimmed_corrections[i+1,j] =
      corrections[i,j]-corrections[i+1,j].

    imagestack2 is a translated version of imagestack which
    satisfies::

      imagestack[f,trimmed_corrections[f,0],trimmed_corrections[f,1],trimmed_corrections[f,1]
         approx
      imagestack2[f,0,0,0]

    The shape of imagestack2 and the value of trimmed_corrections
    depend upon the value of 'mode' supplied.   "valid" insists
    that every value of imagestack2 arises from a value in imagestack.
    When "full" is used, every value in imagestack is placed somewhere
    in imagestack2.
    '''

    if imagestack.shape[0]==1:
        raise ValueError("this imagestack has only one frame; it is meaningless"
                        "to try to register one frame")

    # check for trivial axes
    n=len(imagestack.shape)-1
    newcorr=[]
    expandos=[]
    nonzero_guys=[]
    for i in range(1,n+1):
        if imagestack.shape[i]==1:
            expandos.append(i)
            if not np.allclose(corrections[:,i-1],corrections[0,i-1]):
                raise ValueError(f"imagestack has shape only 1 along dimension {i-1}, yet"
                                    "corrections suggest we wiggle along that dimension")
        else:
            nonzero_guys.append(i)
            newcorr.append(corrections[:,i-1])
    newcorr=np.stack(newcorr,axis=1)

    reg,newt=apply_translation_registration(np.squeeze(imagestack),newcorr,mode,interpolation_method)

    reg=np.expand_dims(reg,tuple(expandos))

    newt2=np.zeros((newt.shape[0],n))
    for i,j in enumerate(nonzero_guys):
        newt2[:,j-1]=newt[:,i]

    return reg,newt2


def calc_valid_region(shp,t,interpolation_method='hermite'):
    '''
    Input
    * shp -- n-vector (int)
    * t   -- kxn (floating)

    Output
    * newt -- kxn-vector floating
    * sz -- n-vector (int)

    Let X be a tensor of shape

        k x M0 x M1 x ... M(n-1)

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

        k x M0 x M1 x ... M(n-1)

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

@dataclasses.dataclass
class EstimateAffineRegistrationsResult:
    estimates: np.ndarray
    residuals: np.ndarray

def estimate_affine_registrations(points,translations):
    '''
    Input:

    - points        -- Q x n
    - translations  -- Q x F x n

    Output:

    - affines_est, F x n x (n+1)
    - variance_unexplained:  scalar indicating poorness of fit

    For each q in 0.... (Q-1), we assume we have attempted
    a translation-based registration effort in a local region
    around points[q].  We assume it yielded proposed
    translation-registrations given by
    translations[q].  This function returns affines_est, which
    indicates a reasonable best-guess affine fit.  To make this
    unique we enforce that (affines_est[0]=0).all().
    '''
    q,F,n=translations.shape
    tcs_with_one = np.c_[points,np.ones(len(points))]
    # ts_recentered = translations-np.mean(translations,axis=1,keepdims=True)
    ts_recentered = translations-translations[:,[0],:]
    affines_est=np.linalg.lstsq(tcs_with_one,ts_recentered.reshape((q,-1)),rcond=None)[0]

    affine_reconstruction = tcs_with_one@affines_est
    residual = affine_reconstruction - ts_recentered.reshape((q,-1))

    affines_est=affines_est.reshape((n+1,F,n))
    affines_est=np.swapaxes(affines_est,0,1)
    affines_est=np.swapaxes(affines_est,2,1)

    resid=np.reshape(residual,(q,F,n))

    return EstimateAffineRegistrationsResult(affines_est,resid)

    # resid2=resid-np.mean(resid,axis=0,keepdims=True)
    # orig2=translations-np.mean(translations,axis=0,keepdims=True)
    # variance_unexplained=np.mean(resid2**2)/np.mean(orig2**2)

    # return affines_est,variance_unexplained


def apply_small_affine_registrations(X,affines,sz=None,constant_values=0,
                                                interpolation_method='nearest'):
    rez=[]
    for f in range(X.shape[0]):
        rez.append(apply_small_affine_registration(X[f],affines[f],sz,constant_values=0,
                                                interpolation_method='nearest'))
    return np.stack(rez,axis=0)


def apply_small_affine_registration(X,affine,sz=None,constant_values=0,
                                                interpolation_method='nearest',
                                                use_tqdm_notebook=False,
                                                out_dtype=None):
    '''
    Input
    - X       -- M0 x M1 x ... M(n-1)
    - affine  -- n x (n+1)
    - sz      -- n
    - constant_values, a scalar

    Output
    - Y       -- sz[0] x sz[1] x ... sz[n-1]

    Roughly speaking, for each m0,m1,m2..., the output satisfies

        ms_with_bias=np.array([m0,m1,m2...,1])
        grabpoint = ms+(affines@ms_with_bias).require(int)
        Y[m0,m1,m2,...] = X[grabpoint]

    For grabpoints out of bounds for X, constant_values are assigned to Y.

    To do this efficiently, we find regions of space where
    (affines@ms_with_bias).require(int) is constant.  This is only
    possible if affines is pretty dang small.  If it isn't this small,
    you'll need to use apply_large_affine_registration
    '''

    if sz is None:
        sz=X.shape

    if out_dtype is None:
        out_dtype = X.dtype

    # send X to tensorflow
    X=tf.identity(X)

    sz=np.require(sz,dtype=int)
    n=len(sz)

    # check if we can get good rectangle
    rect=calc_reasonable_rectangles(affine[:,:n])

    # use those tiles
    tiles=tiling.tile_up_nd(sz,rect)

    Y=np.zeros(X.shape,dtype=out_dtype)

    for tile in misc.maybe_tqdm(tiles,use_tqdm_notebook):
        # get a coordinate in the center of the tile
        lc=tile.look.center.astype(int) # in global coords of Y
        gc=tile.grab.center.astype(int) # in local tile coords

        # find translation for the center
        translation_at_center=affine@np.r_[lc,1] + lc

        # infer what translation should be at beginning
        # of this tile
        translation_at_start = translation_at_center-gc

        # so now Y[f][tile.look][0,0,0]... = X[f][translation_at_start]
        # stick it in
        targ=tile.look.as_slices
        val=translations_tf.floating_slice(
            X,
            translation_at_start,
            tf.cast(tile.look.size,tf.int32),
            interpolation_method=interpolation_method,
            constant_values=constant_values
        )
        Y[targ]=val.numpy()

    return Y



def apply_large_affine_registration(X,affines,sz,mode='valid',constant_values=0,
                                                interpolation_method='nearest'):
    '''
    Input
    - X       -- M0 x M1 x ... M(n-1)
    - affines -- n x (n+1)
    - sz      -- n
    - constant_values, a scalar

    Output
    - Y       -- sz[0] x sz[1] x ... sz[n-1]

    Roughly speaking, for each m0,m1,m2..., the output satisfies

        ms=np.array([m0,m1,m2...,mn,1])
        ms_with_bias=np.r_[ms,1]
        grabpoint = ms+(affines@ms_with_bias).require(int)
        Y[m0,m1,m2,...] = X[grabpoint]

    For grabpoints out of bounds of X, constant_values is assigned to Y.

    This function is pretty inefficient, performing lots of gathers
    to get the desired answers.  Consider using apply_small_affine_registration
    if it meets your needs.
    '''

    sz=np.require(sz,dtype=int)
    m=len(sz)
    n=len(X.shape)
    assert m==n

    # construct massive meshgrid, sz[0] x sz[1] x ... x sz[m-1] x m
    meshgrid=np.stack(np.meshgrid(*[np.r_[0:x] for x in sz],indexing='ij'),axis=-1)

    # apply affine transformations to the meshgrid
    meshgrid = meshgrid + meshgrid @ affines[:,:m].T # sz[0] x sz[1] x ... x sz[m-1] x n
    meshgrid = meshgrid + affines[:,m]

    # sample
    meshgrid = np.reshape(meshgrid,(-1,n))
    Y=translations_tf.sample(X,meshgrid,
        interpolation_method=interpolation_method,constant_values=constant_values).numpy()
    Y=np.reshape(Y,sz)
    return Y

def apply_translation_registration(mini,totalt,mode='valid',interpolation_method='linear'):
    '''
    Input
    - mini -- F x M0 x M1 ... M(n-1)
    - totalt -- F x n
    - interpolation_method (hermite/linear/nearest)
    - mode (valid/full)

    Output
    - minir -- a rigidly registered version of mini
    - t     -- F x n, indicating the cooridnates in mini which are
                    used to supply minir[:,0,0,0,...]
    '''
    totalt=np.require(totalt,dtype=float)
    F=mini.shape[0]
    if mode=='valid':
        newt,sz=calc_valid_region(mini.shape[1:],
            totalt,interpolation_method)
    elif mode=='full':
        newt,sz=calc_complete_region(mini.shape[1:],
            totalt,interpolation_method)
    else:
        raise NotImplementedError(mode)
    minir=translations_tf.floating_slices(mini,
            newt,sz,interpolation_method)
    return minir.numpy(),newt.numpy()

def distributed_translation_estimator(D):
    '''
    Compute registration among many frames,
    based on pairwise estimated registrations.

    - Input: D (... x F x F numpy array)
    - Output: T (... x F numpy array)

    Solves the minimization problem::

        argmin_t sum_ij (D_ij^2 - (t_i-t_j))^2

    Ihe input should indicate the result of trying
    to find a translation that registers each pair of frames.  For
    example, D_ij could represent the result of running
    `pairwise_correlation_registration(data[i],data[j])`.
    The output tries to find a translation for all of the frames
    that is as consistent as possible with all the pairwise
    estimates.

    For more info, cf. E. Varol et al.,
    "Decentralized Motion Inference and Registration of
    Neuropixel Data," ICASSP 2021 - 2021 IEEE
    International Conference on Acoustics,
    Speech and Signal Processing (ICASSP), 2021, pp. 1085-1089,
    doi: 10.1109/ICASSP39728.2021.9414145.
    '''

    D2 = np.mean(D,axis=-2,keepdims=True)
    residual = D-D2
    return np.mean(residual,axis=-1)

def dotproducts_at_translations(X,Y,cutin=None,demean=True):
    '''
    Computes dot product between X and various translations of Y,
    after a little bit of preprocessing.

    Input:

    - X -- M0 x M1 X ... M(n-1) numpy array
    - Y -- M0 x M1 x ... M(n-1) numpy array
    - [optional] cutin -- a scalar or n-vector indicating
      that we should only look at an inner portion of Y;
      this can help especially if X,Y aren't demeaned
    - [optional] demean -- whether or not to subtract mean from X and
      Y before estimating translations

    Output:

    - V -- an n-dimensional numpy array indicating dot product between
      X and various translations of Y.
    - offset -- an n-vector

    The offset indicates how the entries in V should be interpreted:
    V[i0,i1,i2...] indicates the dot product between X and Y after
    translating Y by (i0+offset0,i1+offset1,i2+offset2...).  For example,
    If demean and cutin are False::

        V[i0,i1,i2...] = sum_j X[j0,j1,j2...] * Y[j0-(i0+offset0),j1-(i1+offset1)...].

    Here the sum is taken over all values such that the indices aren't
    out-of-bounds.

    NOTE: this is a thin wrapper around sp.signal.correlate, and it does
    not use GPUs.
    '''

    n=len(X.shape)

    if isinstance(cutin,int):
        cutin_vec=np.ones(n,dtype=int)*cutin
    elif cutin is None:
        cutin_vec=np.zeros(n,dtype=int)
    else:
        cutin_vec=np.require(cutin,dtype=int)
        assert cutin_vec.shape==(n,)

    if demean:
        X=X-np.mean(X)
        Y=Y-np.mean(Y)

    cutin_st=cutin_vec
    cutin_en = np.r_[Y.shape]-cutin_vec
    cutin_slice=tuple([slice(s,e) for (s,e) in zip(cutin_st,cutin_en)])
    Y_cutin=Y[cutin_slice]

    corrs=sp.signal.correlate(X.astype(float),Y_cutin.astype(float),mode='full')
    return corrs,1+cutin_st - np.r_[Y.shape]

def pairwise_correlation_registration(X,Y,cutin=None,demean=True):
    '''
    Estimate a translation that makes X and Y line up with
    each other.

    Input:

    - X -- M0 x M1 X ... M(n-1) numpy array
    - Y -- M0 x M1 x ... M(n-1) numpy array
    - [optional] cutin -- a scalar or n-vector indicating
      that we should only look at an inner portion of Y;
      this can help especially if X,Y aren't demeaned
    - [optional] demean -- whether or not to subtract mean from X and
      Y before estimating translations

    Output: t, an n-vector indicating how Y should be shifted to
    look like X.  Roughly speaking, t tries to make it so that::

        X[i0,i1,i2...] approx Y[i0-t0,i1-t1,...]

    by looking at dot products between X and various translations of
    Y.  Put another way, if we run::

        (Xp,Yp) = apply_translations([X,Y],[zeros(n),-t],'full')

    then Xp and Yp should line up.

    See `dotproducts_at_translations` for more information.

    NOTE: this is a thin wrapper around sp.signal.correlate, and it does
    not use GPUs.
    '''
    corrs,offset=dotproducts_at_translations(
        X,Y,cutin=cutin,demean=demean
    )
    best=np.array(misc.argmax_nd(corrs))
    return best+offset

def erdem_normcorregister(X,submean=True,cutin=None):
    '''
    Input
    * mini -- F x M0 x M1 x M2 x ... M(n-1)

    Output
    * ts -- F x n
    '''

    F=X.shape[0]
    n=len(X.shape)-1
    D=np.zeros((F,F,n))

    if submean:
        for d in range(1,n+1):
            X=X-np.mean(X,axis=d,keepdims=True)

    for f1 in range(F):
        for f2 in range(f1+1,F):
            D[f1,f2]=-normcorrregister_pair(X[f1],X[f2],cutin=cutin)
            D[f2,f1]=-D[f1,f2]

    D2 = np.mean(D,axis=0,keepdims=True)
    residual = D-D2
    return np.mean(residual,axis=1)