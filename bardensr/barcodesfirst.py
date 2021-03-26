import tensorflow as tf
import numpy as np
import pandas as pd


@tf.function
def _build_density(Xshfl,codes,lowg):
    nrms=tf.math.sqrt(tf.reduce_sum(Xshfl**2,axis=0,keepdims=True))
    Xshfl=Xshfl/(lowg+nrms)
    codes=codes/tf.math.sqrt(tf.reduce_sum(codes**2,axis=0,keepdims=True))
    dots=tf.transpose(Xshfl)@codes
    return dots

def build_density_from_tf(Xsh,codebook,lowg_constant):
    R,C,M0,M1,M2=Xsh.shape
    R,C,J=codebook.shape
    Xshfl=tf.reshape(Xsh,(R*C,M0*M1*M2))
    codebook=tf.reshape(codebook,(R*C,J))
    dots = _build_density(Xshfl,codebook,lowg_constant)
    return tf.reshape(dots,(M0,M1,M2,J))

def build_density(Xsh,codebook,lowg_factor=.0005,lowg_constant=None,double_precision=False):
    # Xsh -- R,C,M0,M1,M2

    if lowg_constant is None:
        lowg_constant = float(Xsh.max()*lowg_factor*np.prod(Xsh.shape[:2]))
    else:
        lowg_constant=float(lowg_constant)

    if double_precision:
        dtype=tf.float64
    else:
        dtype=tf.float32

    codebook=codebook.copy()
    codebook[np.isnan(codebook)]=0
    lowg=tf.convert_to_tensor(lowg_constant,dtype=dtype)
    Xshfl=tf.convert_to_tensor(Xsh.reshape((-1,np.prod(Xsh.shape[-3:]))),dtype=dtype)
    codes=tf.convert_to_tensor(codebook.reshape((-1,codebook.shape[-1])),dtype=dtype)
    dots=_build_density(Xshfl,codes,lowg).numpy()
    densities=dots.reshape(Xsh.shape[-3:]+(-1,))

    return densities

@tf.function(autograph=False)
def _sharpen(densities,sharpening_level,sharpening_filter):
    densities=tf.transpose(densities,[3,0,1,2])[...,None] # J X M0 x M1 X M2 x 1
    densities_blurred=tf.nn.convolution(densities,sharpening_filter[...,None,None],padding='SAME')
    print(sharpening_level)
    densities_sharpened = densities + sharpening_level*(densities-densities_blurred)
    densities_sharpened=tf.transpose(densities_sharpened,[1,2,3,0,4])
    return densities_sharpened

def sharpen(densities,sharpening,sigma):
    sharpening=tf.convert_to_tensor(sharpening,dtype=tf.float32)
    filter=np.exp(-.5*np.r_[-5:5]**2/(sigma*sigma))
    print(filter)
    filter=filter[:,None,None]*filter[None,:,None]*filter[None,None,:]
    filter=filter/np.sum(filter)
    print(np.sum(filter))
    filter=tf.convert_to_tensor(filter,dtype=tf.float32)
    densities=tf.convert_to_tensor(densities,dtype=tf.float32)
    sh=_sharpen(densities,sharpening,filter).numpy()[...,0]
    return sh

@tf.function
def _peak_call(densities,poolsize,thresh):
    densities=tf.transpose(densities,[3,0,1,2])[...,None] # J X M0 x M1 X M2 x 1
    mp=tf.nn.max_pool(densities,poolsize,1,'SAME')
    locs=tf.where(mp==densities)
    vals=tf.gather_nd(mp,locs)
    locs=locs[vals>thresh]
    return locs

def peak_call(densities,poolsize,thresh):
    '''
    Input
    - densities, M0 x M1 x M2 x J
    - poolsize, tuple of ints
    - thresh, scalar
    '''
    thresh=tf.convert_to_tensor(thresh,dtype=tf.float32)
    densities=tf.convert_to_tensor(densities,dtype=tf.float32)
    locs=_peak_call(densities,poolsize,thresh).numpy()
    return pd.DataFrame(dict(m0=locs[:,1],m1=locs[:,2],m2=locs[:,3],j=locs[:,0]))
