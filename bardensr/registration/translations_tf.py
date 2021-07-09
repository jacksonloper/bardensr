import tensorflow as tf
import numpy as np

from bardensr import tf_helpers

#######################
# translation kernels



def hermite_interpolation(val1,deriv1,val2,deriv2,t):
    '''
    Input
    - val1
    - deriv1
    - val2
    - deriv2
    - t

    All shapes must be broadcastable to each other.

    val1,deriv1,val2,deriv2 must be same dtype,
    and output will have this dtype.

    t can (and perhaps should) have higher precision.

    Output: hermite cubic interpolation on the unit
    interval.  That is, we imagine

    f(0) = val1
    f'(0) = deriv1
    f(1) = val2
    f'(1) = deriv2

    and we are trying to get a reasonable value for f(t)
    '''

    with tf.name_scope('hermite_interpolation'):
        t2 = t**2
        t3 = t2*t

        h00 = 2*t3 - 3*t2+1
        h10 = t3 - 2*t2 + t
        h01 = -2*t3 + 3*t2
        h11 = t3 - t2

        # h00*val1 + h10*deriv1 + h01*val2 + h11*deriv2
        msm=mixed_precision_multo
        return msm(h00,val1) + msm(h10,deriv1) + msm(h01,val2) + msm(h11,deriv2)


def mixed_precision_multo(a,b):
    return tf.cast(a,dtype=b.dtype)*b

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
    with tf.name_scope("linear_interpolation"):
        msm=mixed_precision_multo

        # val1*(1-t) + val2*t
        return msm(1-t,val1) + msm(t,val2)


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
    return tf.map_fn(
        lambda inp: floating_slice(inp[0],inp[1],sz,interpolation_method,cval),
        (X,t),
        fn_output_signature=X.dtype
    )
    # newXs = [floating_slice(X[f],t[f],sz,interpolation_method,cval) for f in range(X.shape[0])]
    # return tf.stack(newXs,axis=0)

def floating_slice(X,t,sz,interpolation_method,constant_values=0,name=None):
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
    * method -- 'hermite' or 'linear' or 'nearest'
    * cval -- scalar, what to do for oob indexes

    Out:
    * Y -- sz

    '''


    if name is None:
        name='floating_slice'
    with tf.name_scope(name):

        with tf.name_scope("casting"):
            tic=tf.cast(t,dtype=tf.int32)
            sz=tf.cast(sz,dtype=tf.int32)

        '''
        STEP 1.  Big slices.
        - Where t is greater than 2, we can slice into it without
            losing any information for any interpolation we might want
        - Where t+sz is less than shape-2, we can slice into it
            without losing any information for any interpolation we might want
        '''

        with tf.name_scope("bigslices"):

            shp=tf.shape(X)
            left_slicein=tf.clip_by_value(tic-2,0,shp-2)
            right_doable_size=tf.clip_by_value(left_slicein+sz+2,0,shp-left_slicein) # left_slicein+doable_sz <=shp

            X=tf.slice(X,left_slicein,right_doable_size)
            t=t-tf.cast(left_slicein,t.dtype)

        '''
        STEP 2.  Bitty bits.
        '''

        p=t%1
        for d in range(t.shape[0]):
            if interpolation_method=='hermite': # Y=X[p-1:shp+p]
                X = hermite_small_translation_1d_padded(X,d,p[d])
            elif interpolation_method=='linear': # Y=X[p-1:shp+p]
                X = linear_small_translation_1d_padded(X,d,p[d])
            elif interpolation_method=='nearest': # Y=X
                pass
            else:
                raise NotImplementedError()

        '''
        STEP 3.  Final pad and slice.
        '''

        with tf.name_scope("finalpadnslice"):
            if interpolation_method=='hermite' or interpolation_method=='linear':
                ic=tf.cast(tf.math.floor(t),tf.int32)
                # current left edge of X is located at p-1
                # we want the current left edge of X to be at t=ic+p
                left_wrong = ic+1  # (ic+p) - (p-1)

                # current right edge of X is located at shp+p
                # we want it to be at 1+t+sz=1+ic+p+sz
                right_wrong = 1+sz+ic-tf.shape(X) # (ic+p+sz) - (shp+p)
            elif interpolation_method=='nearest':
                ic=tf.cast(tf.math.round(t),tf.int32)
                left_wrong   = ic
                right_wrong  = sz+ic-tf.shape(X)
            else:
                raise NotImplementedError()

            # get pad and slice necessary to make Y what we want
            pad_left   = tf.clip_by_value(-left_wrong,0,left_wrong.dtype.max)
            slice_left = tf.clip_by_value(left_wrong,0,left_wrong.dtype.max)
            pad_right  = tf.clip_by_value(right_wrong,0,left_wrong.dtype.max)

            # do it!
            Y=tf.pad(X,tf.stack([pad_left,pad_right],axis=-1))
            Z=tf.slice(Y,slice_left,sz)

            return Z

def sample(X,points,interpolation_method='nearest',constant_values=0.0):
    '''
    Input
    * X      -- M0 x M1 x M2 ... M(n-1)
    * points -- K x n
    * interpolation_method (only 'nearest' implemented at this time)

    Output
    * Y -- K

    Such that, roughly speaking Y[k]= X[points[k]]

    When interpolation_method is 'nearest', this is a thin
    wrapper around gather which ensures that any attempts to
    access indices which are oob for X yield 0.0.
    '''

    n=len(X.shape)
    K=len(points)

    if interpolation_method=='nearest':
        points=tf.cast(tf.round(points),tf.int32)

        # get good points
        zero=tf.zeros(n,dtype=tf.int32)
        sz=tf.convert_to_tensor(X.shape,dtype=tf.int32)
        good=tf.reduce_all((points>=zero)&(points<sz),axis=-1)

        # do the gather on the good points
        gathered=tf.gather_nd(X,tf.boolean_mask(points,good))

        # scatter the good values out to the right places
        default=tf.fill((K,),tf.cast(constant_values,dtype=X.dtype))
        return tf.tensor_scatter_nd_update(
            default,tf.where(good),gathered,(points.shape[0],))
    else:
        raise NotImplementedError()

def hermite_small_translation_1d(X,axis,t,name=None):
    '''
    Translates X along axis by an amount 0<=t<=1.  Intuitively, something like

        result[...,i...,] approx X[...,1+i+t,...]

    where t is floating point  The shape
    of result will be same as the shape of X,
    except along axis, where it will be reduced by three.


    0 A B C D 0

    '''

    if name is None:
        name='hermite_small_translation_1d'
    with tf.name_scope(name):

        shp=tf.shape(X)

        # get four neighbors
        stv,szv=tf_helpers.construct_1dslice(shp,0,shp[axis]-3,axis)
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

def hermite_small_translation_1d_padded(X,axis,t,constant_values=0.0,name=None):
    '''
    Translates X along axis by an amount 0<=t<=1.  Intuitively, something like

        result[...,i...,] approx X[...,i+t-1,...]

    where t is floating point.  The shape
    of result will be same as the shape of X,
    except along axis, where it will be increased by 1.


        q   A   B   C   D   E   q         t=.2
     q   A'  B'  C'  D'  E'  F'   q
        |   |   |   |   |   |   |
        A*  B*  C*  D*  E*  F*  G*

    We need to compute values and derivatives
    for A*,B*,C*,D*...

    - f(A*) = q, f'(A*)=0    <-- early cases
    - f(B*) = A, f'(B*)=(B-A)

    - f(C*) = B, f'(C*)=(C-A)/2   <-- this is the general case
    - f(D*) = C, f'(D*)=(E-C)/2

    - f(F*) = E, f'(F*) = (E-D)
    - f(G*) = q, f'(G*) = 0 <-- final cases

    '''

    if name is None:
        name='hermite_small_translation_1d_padded'

    with tf.name_scope(name):

        shp=tf.shape(X)
        n=tf.shape(shp)[0]

        ############
        # get allvals
        with tf.name_scope("allvals"):
            pad=tf.ones(2,dtype=tf.int32)[None,:]*tf.one_hot(axis,n,dtype=tf.int32)[:,None]
            allvals=tf.pad(X,pad,constant_values=constant_values)

        ###################
        # get allderivs

        with tf.name_scope("allderivs"):

            with tf.name_scope('secants'):
                # in the middle, we can get secant derivatives
                stv,szv=tf_helpers.construct_1dslice(shp,0,shp[axis]-2,axis)
                W = tf.slice(X,stv,szv)
                stv=tf.tensor_scatter_nd_add(stv,[[axis]],[2])
                E = tf.slice(X,stv,szv)
                mid_derivs=(E-W)*.5

            with tf.name_scope("penultimates"):
                # on the penultimate cases, we can get one-way derivs
                W1=tf.gather(X,[0],axis=axis)
                W2=tf.gather(X,[1],axis=axis)
                west_deriv=W2-W1

                E1=tf.gather(X,[shp[axis]-2],axis=axis)
                E2=tf.gather(X,[shp[axis]-1],axis=axis)
                east_deriv=E2-E1

            with tf.name_scope("stackitup"):
                # on the ultimate cases -- zero
                zero_deriv=tf.zeros_like(east_deriv)

                # concat!
                allderivs=tf.concat(
                    [zero_deriv,west_deriv,mid_derivs,east_deriv,zero_deriv],axis=axis)



        ##############
        # do interpolation
        with tf.name_scope("sliceandinterpolate"):
            stv,szv=tf_helpers.construct_1dslice(shp,0,shp[axis]+1,axis)
            W = tf.slice(allvals,stv,szv)
            Wd = tf.slice(allderivs,stv,szv)
            stv=tf.tensor_scatter_nd_add(stv,[[axis]],[1])
            E = tf.slice(allvals,stv,szv)
            Ed = tf.slice(allderivs,stv,szv)

            # hermite interpolation does the rest
            return hermite_interpolation(W,Wd,E,Ed,t)

def linear_small_translation_1d(X,axis,t,name=None):
    '''
    Translates X along axis by an amount 0<=t<=1.  Intuitively, something like

        result[...,i...,] approx X[...,i+t,...]

    where t is floating point.  The shape
    of result will be same as the shape of X,
    except along axis, where it will reduced by 1.
    '''

    if name is None:
        name='linear_small_translation_1d'

    with tf.name_scope(name):

        shp=tf.shape(X)

        # get two neighbors
        stv,szv=tf_helpers.construct_1dslice(shp,0,shp[axis]-1,axis)
        W = tf.slice(X,stv,szv)

        stv=tf.tensor_scatter_nd_add(stv,[[axis]],[1])
        E = tf.slice(X,stv,szv)

        # linear interpolation does the rest
        return linear_interpolation(W,E,t)


def linear_small_translation_1d_padded(X,axis,t,constant_values=0.0,name=None):
    '''
    Translates X along axis by an amount 0<=t<=1.  Intuitively, something like

        result[...,i...,] approx X[...,i+t,...]

    where t is floating point.  The shape
    of result will be same as the shape of X,
    except along axis, where it will be increased by 1.


        q   A   B   C   D   E   q         t=.2
     q   A'  B'  C'  D'  E'  F'   q
        |   |   |   |   |   |   |
        A*  B*  C*  D*  E*  F*  G*


    '''

    if name is None:
        name='linear_small_translation_1d_padded'

    with tf.name_scope(name):

        shp=tf.shape(X)
        n=tf.shape(shp)[0]

        # pad
        pad=tf.ones(2,dtype=tf.int32)[None,:]*tf.one_hot(axis,n,dtype=tf.int32)[:,None]
        allvals=tf.pad(X,pad,constant_values=constant_values)

        # get two neighbors
        stv,szv=tf_helpers.construct_1dslice(shp,0,shp[axis]+1,axis)
        W = tf.slice(allvals,stv,szv)

        stv=tf.tensor_scatter_nd_add(stv,[[axis]],[1])
        E = tf.slice(allvals,stv,szv)

        # linear interpolation does the rest
        return linear_interpolation(W,E,t)
