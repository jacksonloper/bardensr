from . import translations_tf
import tensorflow as tf
from .. import misc
from ..rectangles import tiling
import scipy as sp
import scipy.ndimage
import numpy as np
import numbers
import typing
from .analytic_centering import calc_reasonable_rectangles
from .lowrankregistration import lowrankregister

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


def estimate_affine_registrations(points,translations):
    '''
    Input:
    - points        -- Q x n
    - translations  -- Q x F x n

    Output:
    - affines_est, F x n x (n+1)
    - proportion_variance_unexplained, scalar

    For each q in 0.... (Q-1), we assume we have attempted
    a translation-based registration effort.  We assume
    this effort was focused on indices centered at points[q],
    and yielded proposed translation-registration given by
    translations[q].  We return

        affines_est, F x n x (n+1)

    Indicating a reasonable best-guess affine fit.  To make this
    unique we use the coordinate system of first frame
    (i.e. affines_est[0]=0).
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
    return affines_est, residual


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

        ms=np.array([m0,m1,m2...,mn,1])
        ms_with_bias=np.r_[ms,1]
        grabpoint = ms+(affines@ms_with_bias).require(int)
        Y[m0,m1,m2,...] = X[grabpoint]

    For grabpoints out of bounds of X, constant_values is assigned to Y.

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

    targets=[]
    values=[]

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
        targets.append(tile.look.as_slices)
        values.append(translations_tf.floating_slice(
            X,
            translation_at_start,
            tf.cast(tile.look.size,tf.int32),
            interpolation_method=interpolation_method,
            constant_values=constant_values
        ))

    Y=np.zeros(X.shape,dtype=out_dtype)
    for t,v in zip(misc.maybe_tqdm(targets,use_tqdm_notebook),values):
        Y[t]=v.numpy()

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
    - mode (valid/complete)

    Output
    - minir -- a rigidly registered version of mini
    - t     -- F x n, indicating the cooridnates in mini which are
                    used to supply minir[:,0,0,0,...]
    '''
    F=mini.shape[0]
    if mode=='valid':
        newt,sz=calc_valid_region(mini.shape[1:],
            totalt,interpolation_method)
    elif mode=='complete':
        newt,sz=calc_complete_region(mini.shape[1:],
            totalt,interpolation_method)
    else:
        raise NotImplementedError(mode)
    minir=translations_tf.floating_slices(mini,
            newt,sz,interpolation_method)
    return minir.numpy(),newt.numpy()

def normcorrregister_pair(X,Y,cutin=None):
    '''
    Input
    * X -- M0 x M1 X ... M(n-1)
    * Y -- M0 x M1 x ... M(n-1)
    '''

    n=len(X.shape)

    if isinstance(cutin,int):
        cutin_vec=np.ones(n,dtype=int)*cutin
    elif cutin is None:
        cutin_vec=np.zeros(n,dtype=int)
    else:
        cutin_vec=np.require(cutin,dtype=int)
        assert cutin_vec.shape==(n,)

    cutin_st=cutin_vec
    cutin_en = np.r_[Y.shape]-cutin_vec
    cutin_slice=tuple([slice(s,e) for (s,e) in zip(cutin_st,cutin_en)])
    Y_cutin=Y[cutin_slice]

    corrs=sp.signal.correlate(X.astype(float),Y_cutin.astype(float),mode='full')
    best=np.array(misc.argmax_nd(corrs))
    return np.r_[Y.shape]-best-1-cutin_st

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