__all__=[
    'estimate_density_iterative',
    'estimate_density_singleshot',
    'find_peaks',
]

import tensorflow as tf
import pandas as pd
import numpy as np
from . import barcodesfirst
from . import blackberry

def estimate_density_iterative(imagestack,codebook,l1_penalty=0,psf_radius=(0,0,0),
                    iterations=100,estimate_codebook_gain=True,
                    rounds=None,
                    estimate_colormixing=False, estimate_phasing=False,
                    use_tqdm_notebook=False):
    '''
    An optimization-based approach estimate the density.

    Input:

    - imagestack (N x M0 x M1 x M2 numpy array)
    - codebook (N x J numpy array)
    - [optional] l1_penalty (a penalty which sends spurious
      noise signals to zero; this should be higher if the
      noise-level is higher)
    - [optional] psf_radius (a tuple of 3 numbers; default
      (0,0,0); your estimate of the psf magnitude in units
      of voxels for each spatial axis of the imagestack
      (shape of psf is assumed to be Gaussian))
    - [optional] iterations (number of iterations to train;
      default is 100)
    - [optional] estimate_codebook_gain (boolean; default
      True; if True, we will attempt to correct the codebook
      for any per-frame gains, e.g. if frame 4 of the imagestack
      is 10 times brighter than all other frames)
    - [optional] rounds (integer default None; if provided, must
      divide evenly into N, and it is then assumed that the
      frames can be understood as R rounds of imaging with N/C channels per round)
    - [optional] estimate_colormixing (boolean; default False;
      if True, we will attempt to correct the codebook for
      color bleed between channels; only works if "rounds" is provided)
    - [optional] estimate_phasing (boolean; default False; if
      True, we will attempt to correct the codebook for
      incomplete clearing of signal between rounds; only works
      if "rounds" is provided)

    Output is an evidence_tensor (M0 x M1 x M2 x J), an estimate for the
    density giving rise to this imagestack
    '''

    F=imagestack.shape[0]
    J=codebook.shape[-1]
    niter=iterations

    if rounds is None:
        rounds=1
    else:
        assert F%rounds==0
    imagestack_rc = imagestack.reshape((rounds,F//rounds,)+imagestack.shape[1:])
    codebook = codebook.reshape((rounds,F//rounds,codebook.shape[-1]))

    imagestack_rct=tf.convert_to_tensor(np.transpose(imagestack_rc,[2,3,4,0,1])) # m0,m1,m2,r,c

    M0,M1,M2=imagestack_rct.shape[:3]

    m=blackberry.denselearner.Model(codebook,(M0,M1,M2),
            lam=l1_penalty,
            blur_level=psf_radius)

    if use_tqdm_notebook:
        import tqdm.notebook
        t=tqdm.notebook.trange(niter)
    else:
        t=range(niter)
    for i in t:
        m.update_F(imagestack_rct)
        if estimate_codebook_gain:
            m.update_alpha(imagestack_rct)
        if estimate_colormixing:
            m.update_varphi(imagestack_rct)
        if estimate_phasing:
            m.update_rho(imagestack_rct)
        m.update_a(imagestack_rct)
        m.update_b(imagestack_rct)

    return m.F_scaled(),dict(frame_gains=m.alpha.numpy().ravel())

def estimate_density_singleshot(
        imagestack,codebook,noisefloor):
    '''
    A correlation-based approach to get a crude estimate of the density.
    Fast and adequate for many purposes.  Does not account for a point-spread function.

    Input:

    - imagestack (N x M0 x M1 x M2 numpy array)
    - codebook (N x J numpy array)
    - noisefloor (a floating point number indicating your
      estimate for what level of signal is "noise")

    Output: evidence_tensor (M0 x M1 x M2 x J), a crude estimate for the density
    '''

    return barcodesfirst.build_evidence_tensor(
        imagestack,codebook,lowg_constant=noisefloor,double_precision=False)


def load_example(name='ab701a5a-2dc3-11eb-9890-0242ac110002'):
    # load data (including the barcode table B)
    import pkg_resources
    import h5py
    from . import benchmarks
    DATA_PATH = pkg_resources.resource_filename('bardensr.benchmarks', f'{name}.hdf5')

    return benchmarks.load_h5py(DATA_PATH)

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
    mp=tf.nn.max_pool(densities,poolsize,1,'SAME')  # take the max in the window of poolsize.
    locs=tf.where((mp==densities)&(mp>thresh))
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

def find_peaks(densities,thresh,poolsize=(1,1,1)):
    '''
    Find bumps in an evidence tensor.

    Input:

    - evidence tensor (M0 x M1 x M2 x J)
    - threshold
    - [optional] poolsize (tuple of 3 numbers; default (1,1,1); indicating the minimum possible size of a bump)

    Output: bumps, a pandas dataframe with the following columns

    - m0 -- where the bumps were found along the first spatial dimension
    - m1 -- where the bumps were found along the second spatial dimension
    - m2 -- where the bumps were found along the third spatial dimension
    - j -- where the bumps were found along the gene dimension
    - magnitude -- value of evidence_tensor in the middle of the bump
    '''

    poolsize = tuple([int(x*2+1) for x in poolsize])

    thresh=tf.convert_to_tensor(thresh,dtype=tf.float32)
    densities=tf.convert_to_tensor(densities,dtype=tf.float32)
    locs=_peak_call(densities,poolsize,thresh).numpy()

    dct={}
    for i in range(locs.shape[1]-2):
        dct[f'm{i}']=locs[:,i+1]
    dct['j']=locs[:,0]

    return pd.DataFrame(dct)