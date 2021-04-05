import tensorflow as tf
import numpy as np

#######################
# translation kernels

def construct_1dslice(shp,st,sz,axis):
    '''
    Input:
    shp -- result of tf.shape
    st -- scalar tf.int32
    sz -- scalar tf.int32
    axis -- scalar tf.int32

    Output:
    stv
    szv

    Such that
    stv[i] = st if i==axis else 0
    szv[i] = -1 if i==axis else shp[i]
    '''

    nd=tf.shape(shp)

    # make stv
    stv=tf.scatter_nd([[axis]],[st],nd)

    # make szv
    szv=tf.cast(tf.fill(nd,-1),dtype=tf.int32)
    szv=tf.tensor_scatter_nd_update(szv,[[axis]],[sz])

    return stv,szv

def hermite_interpolation(val1,deriv1,val2,deriv2,t):
    '''
    Input
    - val1
    - deriv1
    - val2
    - deriv2
    - t

    All shapes must be broadcastable to each other.

    Output: hermite cubic interpolation on the unit
    interval.  That is, we imagine

    f(0) = val1
    f'(0) = deriv1
    f(1) = val2
    f'(1) = deriv2

    and we are trying to get a reasonable value for f(t)
    '''

    t2 = t**2
    t3 = t2*t

    h00 = 2*t3 - 3*t2+1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2

    return h00*val1 + h10*deriv1 + h01*val2 + h11*deriv2


def linear_interpolation(val1,val2,t):
    '''
    Input
    - val1
    - val2
    - t

    All shapes must be broadcastable to each other.

    Output: linear interpolation on the unit
    interval.  That is, we imagine

    f(0) = val1
    f(1) = val2

    and we are trying to get a reasonable value for f(t)
    '''

    return val1*(1-t) + val2*t



def floating_slices(X,t,sz,interpolation_method,cval=0):
    '''
    Batch version of floating_slice, where all the calls
    have the same value for "sz."

        result[k] = X[k,t[k,0]:t[k,0]+sz[0], t[k,1]:t[k,1]+sz[1], ...]

    Input:
    * X -- K x M0 x M1 x M2 ... x Mn
    * t -- K x n floating point
    * sz -- n integer

    Out:
    * Y -- K x [[sz]]

    '''
    sz=tf.convert_to_tensor(sz)
    newXs = [floating_slice(X[f],t[f],sz,interpolation_method,cval) for f in range(X.shape[0])]
    return tf.stack(newXs,axis=0)

def floating_slice(X,t,sz,interpolation_method,cval=0):
    '''
    This function interpolates a slice of X
    at floating-point locations.  That is,

        result = X[t[0]:t[0]+sz[0], t[1]:t[1]+sz[1], ...]

    where

    - t is floating point and interpolation is performed
      via interpolation_method (hermite or linear).
    - if the relevant indices are out of bounds for X (or,
      more specifically, outside the set of locations which can
      be correctly interpolated via interpolation_method) we give
      the value cval

    Input:
    * X -- M0 x M1 x M2 ... x Mn
    * t -- n floating point
    * sz -- n integer

    Out:
    * Y -- sz

    '''
    X=tf.convert_to_tensor(X)
    t=tf.cast(tf.convert_to_tensor(t),dtype=X.dtype)
    sz=tf.convert_to_tensor(sz)

    assert X.dtype==tf.float32 or X.dtype==tf.float64
    assert t.dtype==X.dtype
    assert sz.dtype==tf.int32 or sz.dtype==tf.int64

    p=t%1

    # perform bitty bit
    Y=X
    for d in range(t.shape[0]):
        if interpolation_method=='hermite': # Y=X[1+p:-2+p]
            Y = hermite_small_translation_1d(Y,d,p[d])
        elif interpolation_method=='linear': # Y=X[p:-1+p]
            Y = linear_small_translation_1d(Y,d,p[d])
        else:
            raise NotImplementedError()

    # how different is Y from the slice we want?
    ic=tf.cast(tf.math.floor(t),tf.int32)
    if interpolation_method=='hermite':
        left_wrong   = ic-1
        right_wrong  = sz+ic-1-tf.shape(Y)
    elif interpolation_method=='linear':
        left_wrong   = ic
        right_wrong  = sz+ic-tf.shape(Y)
    else:
        raise NotImplementedError()

    # get pad and slice necessary to make Y what we want
    pad_left   = tf.clip_by_value(-left_wrong,0,left_wrong.dtype.max)
    slice_left = tf.clip_by_value(left_wrong,0,left_wrong.dtype.max)
    pad_right  = tf.clip_by_value(right_wrong,0,left_wrong.dtype.max)

    # do it!
    Z=tf.pad(Y,tf.stack([pad_left,pad_right],axis=-1))
    Z=tf.slice(Z,slice_left,sz)

    return Z

def hermite_samples_1d(X,axis,ts):
    '''
    Samples X along axis.  Intuitively, something like

        result[...,i...,] approx X[...,ts[i],...]

    where t[i] may be floating point.  The shape
    of result will be same as the shape of X,
    except along axis, where it will have same dimension as ts

    The user must guarantee that 1 <= floor(ts[i]) < X.shape[i]-2

    Input:
    - X: M0 x M1 x M2 ... Mn
    - axis: scalar
    - ts: K

    Output:
    - Y: M0 x M1 x ... M(axis-1) x K x M(axis+1) x ... Mn
    '''
    shp=tf.shape(X)
    tshp=tf.shape(ts)

    # get the four neighbors. these are of shape M0 x ... K ... Mn
    integer_component=tf.cast(tf.math.floor(ts),tf.int32)
    W2 = tf.gather(X,integer_component-1,axis=axis) #
    W1 = tf.gather(X,integer_component,axis=axis)
    E1 = tf.gather(X,integer_component+1,axis=axis)
    E2 = tf.gather(X,integer_component+2,axis=axis)

    # estimate derivatives at W1 and E1
    W1_deriv = (E1-W2)/2.0
    E1_deriv = (E2-W1)/2.0

    # broadcast ts correctly
    n=shp.shape[0] #
    newshape=tf.ones(n,dtype=tf.int32)
    newshape=tf.tensor_scatter_nd_update(newshape,[[axis]],tshp)
    props_broadcastable = tf.reshape(ts%1,newshape)

    # hermite interpolation does the rest
    return hermite_interpolation(W1,W1_deriv,E1,E1_deriv,props_broadcastable)

def hermite_small_translation_1d(X,axis,t):
    '''
    Translates X along axis by an amount 0<=t<=1.  Intuitively, something like

        result[...,i...,] approx X[...,1+i+t,...]

    where t is floating point  The shape
    of result will be same as the shape of X,
    except along axis, where it will be reduced by three.
    '''
    shp=tf.shape(X)

    # get four neighbors
    stv,szv=construct_1dslice(shp,0,shp[axis]-3,axis)
    W2 = tf.slice(X,stv,szv)

    stv=tf.tensor_scatter_nd_add(stv,[[axis]],[1])
    W1 = tf.slice(X,stv,szv)

    stv=tf.tensor_scatter_nd_add(stv,[[axis]],[1])
    E1 = tf.slice(X,stv,szv)

    stv=tf.tensor_scatter_nd_add(stv,[[axis]],[1])
    E2 = tf.slice(X,stv,szv)

    # estimate derivatives at W1 and E1
    W1_deriv = (E1-W2)/2.0
    E1_deriv = (E2-W1)/2.0

    # hermite interpolation does the rest
    return hermite_interpolation(W1,W1_deriv,E1,E1_deriv,t)


def linear_small_translation_1d(X,axis,t):
    '''
    Translates X along axis by an amount 0<=t<=1.  Intuitively, something like

        result[...,i...,] approx X[...,i+t,...]

    where t is floating point  The shape
    of result will be same as the shape of X,
    except along axis, where it will be reduced by 1.
    '''
    shp=tf.shape(X)

    # get two neighbors
    stv,szv=construct_1dslice(shp,0,shp[axis]-1,axis)
    W = tf.slice(X,stv,szv)

    stv=tf.tensor_scatter_nd_add(stv,[[axis]],[1])
    E = tf.slice(X,stv,szv)

    # linear interpolation does the rest
    return linear_interpolation(W,E,t)

###################
# blur kernels

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


@tf.function(autograph=False)
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
