import numpy as np
import scipy as sp
import scipy.spatial.distance
from ... import blur_kernels as kernels
import itertools
import numpy.random as npr

import collections


def downsample_nd(X,ns):
    for (i,n) in enumerate(ns):
        X=downsample(X,n,axis=i)
    return X

def downsample(X,n,axis=0):
    shp=list(X.shape)

    # get divisible by n
    slices=[slice(0,None) for i in range(len(shp))]
    slices[axis]=slice(0,n*(shp[axis]//n))
    X=X[tuple(slices)]

    # reshape
    newshp=list(X.shape)
    newshp.insert(axis+1,n)
    newshp[axis]=newshp[axis]//n
    X=np.reshape(X,newshp)

    # average
    X=np.mean(X,axis=axis+1)

    # done!
    return X
