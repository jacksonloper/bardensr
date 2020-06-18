

import numpy as np
import scipy as sp
import scipy.spatial

def pointset_agreement(A,B,closeness):
    '''
    How many of the points in A have a point in B which
    is within "closeness" of it?
    '''

    if len(A)==0:
        return 0
    if len(B)==0:
        return 0

    dst=sp.spatial.distance.cdist(A,B)
    dst = np.min(dst,axis=1)
    return np.sum(dst<=closeness)
