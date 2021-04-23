import tensorflow as tf
import pandas as pd
from . import barcodesfirst

def _peak_call(densities,poolsize,thresh):
    '''
    Input
    - densities, M0 x M1 x ... M(n-1) x J
    - poolsize, tuple of ints
    - thresh, scalar

    Output
    - peaks, n_peaks x (1+n)
    '''

    transposal = tf.concat([
        [len(densities.shape)-1],
        tf.range(len(densities.shape)-1)
    ],0)
    densities=tf.transpose(densities,transposal)[...,None] # J X M0 x M1 X ... x M(n-1) x 1
    mp=tf.nn.max_pool(densities,poolsize,1,'SAME')
    locs=tf.where(mp==densities)
    vals=tf.gather_nd(mp,locs)
    locs=locs[vals>thresh]
    return locs

def peak_call(densities,poolsize,thresh):
    '''
    Input
    - densities, M0 x M1 x ... x J
    - poolsize, tuple of ints
    - thresh, scalar
    '''
    thresh=tf.convert_to_tensor(thresh,dtype=tf.float32)
    densities=tf.convert_to_tensor(densities,dtype=tf.float32)
    locs=_peak_call(densities,poolsize,thresh).numpy()

    dct={}
    for i in range(locs.shape[1]-2):
        dct[f'm{i}']=locs[:,i+1]
    dct['j']=locs[:,0]

    return pd.DataFrame(dct)