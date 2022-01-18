import skimage
import skimage.draw
import skimage.feature
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import numpy.random as npr

from .. import misc
from . import purepixel
from .. import spot_calling

import logging
logger=logging.getLogger(__name__)


############

### merging the barcodes part.
def merge_barcodes(A,B):
    B=B.copy()  # start with B's version (this is arbitrary!)
    B[np.isnan(B)]=A[np.isnan(B)] # but if B is ignorant, use info from A
    return B

def codebook_deduplication_iteration(cb,thre=0,use_tqdm_notebook=False):
    R,C,J=cb.shape

    if J==0:
        return np.zeros((R,C,0),dtype=np.float)

    cb=cb.reshape((R*C,-1))

    new_cbs=cb[:,[0]].copy()
    for i in misc.maybe_tqdm(range(1,cb.shape[-1]),use_tqdm_notebook,leave=False):
        dists=misc.nan_robust_hamming(cb[:,[i]],new_cbs).ravel() # distance to existing codes
        if (dists>thre).all():
            new_cbs=np.concatenate([new_cbs,cb[:,[i]]],axis=1)
        else:
            best=np.argmin(dists)
            new_cbs[:,best] = merge_barcodes(new_cbs[:,best],cb[:,i])
    return new_cbs.reshape((R,C,-1))

def codebook_deduplication(cb,thre=0,use_tqdm_notebook=False):
    R,C,J=cb.shape
    improved=True
    while improved:
        cb=codebook_deduplication_iteration(cb,thre,use_tqdm_notebook)
        if cb.shape[-1]<J:
            improved=True
            J=cb.shape[-1]
        else:
            improved=False
    return cb

def merge_codebooks(book1,book2,thre=0,use_tqdm_notebook=False):
    '''
    TODO: use ratios if you have them?
    '''
    book=np.concatenate([book1,book2],axis=-1)
    book=codebook_deduplication(
        book,thre=thre,use_tqdm_notebook=use_tqdm_notebook)
    return book

#################


def get_denselearner_reconstruction(X,codebook,blob_radius=None,blur_level=0,
                                    n_unused_barcodes=2,bardensr_spot_thresh_multiplier=1.0,niter=120,lam=.01,
                                    return_params = False
                                   ):
    # add extra barcodes
    R,C,J=codebook.shape
    unused_barcodes=misc.convert_codebook_to_onehot_form(npr.randint(0,C,size=(n_unused_barcodes,R)))
    estimated_plus_phony = np.concatenate([codebook,unused_barcodes],axis=-1)

    # run bardensr
    estimated_plus_phony[np.isnan(estimated_plus_phony)]=0.0
    bdresult=spot_calling.blackberry.denselearner.build_density(X,estimated_plus_phony,use_tqdm_notebook=False,
                                                              niter=niter,lam=lam,blur_level=blur_level)

    # get the reconstruction
    RD=bdresult.reconstruction_density.copy()

    # optionally mask out Fs which are below some threshold (lam OUGHT to do this, but... yknow...)
    if blob_radius is not None:
        # get spots we consider "good enough", use this to form a mask
        thresh=bdresult.density[:,:,:,-n_unused_barcodes:].max()*bardensr_spot_thresh_multiplier
        mask=bdresult.density>thresh

        # dilate the mask
        newmask=[]
        for j in range(mask.shape[-1]):
            m=ellipsoid_dilation(mask[:,:,:,j],blob_radius)
            newmask.append(m)
        newmask=np.stack(newmask,axis=-1)
        RD[~newmask]=0

    # done!
    recon = np.einsum('xyzj,rcj->rcxyz',RD,bdresult.reconstruction_codebook)

    if return_params:
        return(recon, RD, bdresult.reconstruction_codebook)
    else:
        return recon

