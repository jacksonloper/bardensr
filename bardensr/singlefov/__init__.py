r'''
This module is designed for handling an observation tensor $\mathbf{X}$ which has the following *properties*...

- **TENSOR STRUCTURE:** The observations can be understood as a tensor $\mathbf{X} \in \mathbb{R}^{M_1 \times M_2 \times M_3 \times R \times C}$.
- **NON-NEGATIVITY:** Each observation is non-negative (i.e. $\mathbf{X}_{m_1,m_2,m_3,r,c}\geq 0$)

... and which can be modeled using the following *parameters*...
- $\mathbf{B} \in \mathbb{R}^{R\times C\times J}$ is called the codebook
- $\rho \in \mathbb{R}^C$ is called the phasing parameter
- $\alpha \in \mathbb{R}^{R\times C}$ is called the gain
- $\varphi \in \mathbb{R}^{C\times C}$ is called the color-mixing
- $\mathbf{F} \in \mathbb{R}^{M_1 \times M_2 \times M_3 \times J}$ is called the density
- $\mathbf{K} \in \mathbb{R}^{M_1 \times M_2 \times M_3 \times M_1 \times M_2 \times M_3}$ is called the kernel, and it is assumed to be a 3-dimensional blur kernel.
- $a \in \mathbb{R}^{M_1 \times M_2 \times M_3}$ is called the spatial background
- $b \in \mathbb{R}^{R\times C}$ is called the frame background

... through the following *story*:

1. **PHASING:** The phasing parameter defines a tensor $\mathbf{Z}$ by letting $$\mathbf{Z}_{r,c,j} = \rho_c \mathbf{Z}_{r-1,c,j} + \mathbf{B}_{r,c,j}$$ for $r\geq 1$.  For $r=0$ we let $\mathbf{Z}_{0,c,j}=\mathbf{B}_{0,c,j}$.
1. **GAIN AND COLOR-MIXING** The gain and color-mixing parameters then give rise the tensor $$\mathbf{G}_{r,c,j} = \alpha_{r,c} \sum_{c'} \varphi_{c,c'}Z_{r,c',j}$$
1. **OBSERVATION MODEL** The observations $\mathbf{X}$ are approximately given by
$$\mathbf{X}_{m_1,m_2,m_3,r,c} \approx \sum_{m'_1=0}^{M_1-1}\sum_{m'_2=0}^{M_2-1}\sum_{m'_1=0}^{M_3-1} \sum_{j=0}^{J-1} \mathbf{G}_{r,c,j} \mathbf{F}_{m'_1,m'_2,m'_3, j} \mathbf{K}_{m_1,m_2,m_3,m'_1,m'_2,m'_3} + a_{m_1,m_2,m_3} + b_{r,c}$$

All the concepts above are described in more depth in the paper ((link forthcoming!)).
'''

__all__=['simulate','process','sparse2dense']

from .simulation import simulate
from . import simulation
from . import processing
from . import denselearner
from . import training
from . import tiling

import numpy as np

def maybe_tqdm(x,tqdm_notebook,**kwargs):
    if tqdm_notebook:
        import tqdm.notebook
        return tqdm.notebook.tqdm(x,**kwargs)
    else:
        return x

def process(
    X,
    B,
    downsample_level=(10,10,2),
    tile_size=(200,200,10),
    phase_I_lambda=.01,
    phase_II_lambda=.01,
    phase_II_learn=(),
    n_unused_barcodes=4,
    unused_barcodes=None,
    unused_barcode_threshold_multiplier=1.0,
    unused_barcode_percentile_by_voxel=1.0,
    unused_barcode_percentile_by_code=1.0,
    tqdm_notebook=False,
    tqdm_notebook_bytile=False,
    blur_level=(3,3,0),
):
    r'''
    Given an observation tensor X and a codebook B, try to guess F.

    Input:
    - X
    - B
    - blur_level=3
    - downsample_level=(10,10,2)
    - tile_size=(200,200,10)
    - phase_I_lambda=.01
    - phase_II_learn=()
    - phase_II_lambda=.1
    - n_unused_barcodes=4
    - unused_barcodes=None
    - unused_barcode_threshold_multiplier=1.0
    - unused_barcode_percentile_by_voxel=100
    - unused_barcode_percentile_by_code=100
    - tqdm_notebook=True

    Output: a sparse representation of where rolony densities are significant
    - vals  -- each entry represents the estimated density present at a particular place --
    - m1s   -- these represent the relevant m1 location
    - m2s   -- these represent the relevant m2 location
    - m3s   -- these represent the relevant m3 location
    - bcds -- these represent the relevant barcode
    That is, for each i, we have that the density at position m1s[i],m2s[i],m3s[i]
    corresponding to barcode bcds[i] has activity level vals[i].  The largest
    barcode indices will correspond to the unused barcodes.

    This algorithm proceeeds in two phases.
    - Phase I.  Downsample and learn F,alpha,varphi,rho,a,b on downsampled data.
    - Phase II.  Break into tiles.  For each tile, unused barcodes give threshold
    to discern codes which can be ignored.  Run on the reduced
    set of codes for this tile, thereby learning F,a (and, optionally, alpha,varphi,rho,b).
    We then stitch tiles back together, store resulting F as a sparse matrix.

    A note on the "unused_barcodes."  We train the model in phase I and phase II as if
    the unused barcodes were present in the data.  We then estimate the density F.  We then
    compute the unused_barcode_threshold by
    - computing the `unused_barcode_percentile_by_voxel` percentile of the density values
    over voxels for each unused barcode
    - computing the `unused_barcodew_percentile_by_code` percentile over the resulting values
    found for each unused barcode
    - multiplying by `unused_barcode_threshold_multiplier`
    We then use this threshold as a way to help guess where density activity is high
    enough that it can be reliably understood.
    '''

    # DOWNSAMPLE!
    Xs = processing.downsample_nd(X,downsample_level)

    # MAKE FAKE BARCODES!
    R,C,J=B.shape
    if unused_barcodes is None:
        B_supplemented=B.copy()
        for i in range(n_unused_barcodes):
            newcode=simulation.sim_onehot_code_against_unknown_codes(B_supplemented)
            B_supplemented=np.concatenate([B_supplemented,newcode[:,:,None]],axis=-1)
    else:
        n_unused_barcodes=len(unused_barcodes)
        B_supplemented=np.concatenate([B,unused_barcodes],axis=-1)

    #####################
    # PHASE I!
    if tqdm_notebook:
        print("Phase I")

    model=denselearner.Model(B_supplemented,
                Xs.shape[:3],
                lam=phase_I_lambda,
                blur_level=[bl//dl for (bl,dl) in zip(blur_level,downsample_level)],
    )
    trainer=training.Trainer(Xs,model)
    trainer.train(['F'],10,tqdm_notebook=tqdm_notebook)
    trainer.train(['alpha','varphi','rho','a','b','F'],20,tqdm_notebook=tqdm_notebook)
    F = model.F_scaled()


    #####################
    # PHASE II!

    if tqdm_notebook:
        print("Phase II")

    # get threshold
    Funused = F[:,:,:,-n_unused_barcodes:]
    Funused = Funused.reshape((-1,n_unused_barcodes))
    thresh = np.percentile(Funused,unused_barcode_percentile_by_voxel,axis=0)
    thresh = np.percentile(thresh,unused_barcode_percentile_by_code,axis=0)
    thresh = thresh*unused_barcode_threshold_multiplier

    # tile it up!
    blx3 = [x*3+1 for x in blur_level]
    tiles = tiling.tile_up_nd(X.shape[:3],tile_size,blx3)

    # process each tile
    Fs=[]
    codes=[]
    trange=tiles
    for tile in maybe_tqdm(tiles,tqdm_notebook):
        dtile = tiling.downsample_multitile(tile,downsample_level)
        Xsub = X[tile.look] # get sub-data
        Fsub = F[dtile.look] # get sub-densities
        Fsubrav = np.reshape(Fsub,(-1,J+n_unused_barcodes))
        good=np.max(Fsubrav,axis=0)>thresh # get codes with promise
        good[-n_unused_barcodes:]=True # always keep the unused barcodes

        # train on the little tile!
        B_little = B_supplemented[:,:,good]
        model2=denselearner.Model(B_little,Xsub.shape[:3],lam=phase_II_lambda,blur_level=blur_level)
        trainer=training.Trainer(Xsub,model2)

        trainer.train(['F'],10,tqdm_notebook=tqdm_notebook_bytile)
        nms=phase_II_learn + ('a','F')
        trainer.train(nms,20,tqdm_notebook=tqdm_notebook_bytile)

        # save it
        codes.append(np.where(good)[0])
        Fs.append(model2.F_scaled()[tile.grab])

    # stitch it together as a sparse matrix
    M1s=[]
    M2s=[]
    M3s=[]
    genes=[]
    values=[]
    for i,tile in enumerate(tiles):
        tile=tiles[i]
        M1sub,M2sub,M3sub,genesub=np.where(Fs[i]>0)
        vsub=Fs[i][M1sub,M2sub,M3sub,genesub]
        M1sub+=tile.put[0].start
        M2sub+=tile.put[1].start
        M3sub+=tile.put[2].start
        genesub=codes[i][genesub]
        M1s.append(M1sub)
        M2s.append(M2sub)
        M3s.append(M3sub)
        genes.append(genesub)
        values.append(vsub)

    return dict(
        values=np.concatenate(values),
        m1=np.concatenate(M1s),
        m2=np.concatenate(M2s),
        m3=np.concatenate(M3s),
        j=np.concatenate(genes)
    )

def sparse2dense(m1,m2,m3,j,values=None,shape=None):
    if shape is None:
        shape=[np.max(x)+1 for x in [m1,m2,m3,j]]
    if values is None:
        values=np.ones(len(m1),dtype=np.float64)
    V=np.zeros(shape,dtype=values.dtype)
    V[m1,m2,m3,j]=values
    return V
