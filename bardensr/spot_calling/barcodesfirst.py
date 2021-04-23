import tensorflow as tf
import numpy as np
import pandas as pd


@tf.function
def _build_evidence_tensor(Xshfl,codes,lowg):
    '''
    Input
    - Xshfl, N x M
    - codebook N x J

    Output[m,j] = <Xshfl[m] | codes[:,j]> / (||norm(Xshfl[m])|| + lowg)



    '''
    nrms=tf.math.sqrt(tf.reduce_sum(Xshfl**2,axis=0,keepdims=True))
    Xshfl=Xshfl/(lowg+nrms)
    codes=codes/tf.math.sqrt(tf.reduce_sum(codes**2,axis=0,keepdims=True))
    dots=tf.transpose(Xshfl)@codes
    return dots

def build_evidence_tensor(Xsh,codebook,lowg_factor=.0005,lowg_constant=None,double_precision=False):
    '''
    Input
    - codebook -- N0,N1,...,J
    - Xsh -- N0,N1,...,M0,M1,...

    Output
    - evidence_tensor -- M0,M1,...,J
    '''


    if double_precision:
        dtype=tf.float64
    else:
        dtype=tf.float32


    codebook=np.require(codebook).copy()
    codebook[np.isnan(codebook)]=0

    Xsh=np.require(Xsh)

    J=codebook.shape[-1]
    nNs=len(codebook.shape)-1
    nMs=len(Xsh.shape) -nNs

    if lowg_constant is None:
        lowg_constant = float(Xsh.max()*lowg_factor*np.prod(Xsh.shape[:nNs]))
    else:
        lowg_constant=float(lowg_constant)

    total_frames=np.prod(codebook.shape[:-1])
    total_vox = np.prod(Xsh.shape[nNs:])

    lowg=tf.cast(tf.identity(lowg_constant),dtype=dtype)
    Xshfl=tf.cast(tf.identity(Xsh.reshape((total_frames,total_vox))),dtype=dtype)
    codes=tf.cast(tf.identity(codebook.reshape((total_frames,J))),dtype=dtype)
    dots=_build_evidence_tensor(Xshfl,codes,lowg).numpy()
    densities=dots.reshape(Xsh.shape[nNs:]+(J,))

    return densities

