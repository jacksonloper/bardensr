import skimage
import skimage.draw
import skimage.feature
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import numpy.random as npr

from .. import misc
from .. import singlefov

import logging
logger=logging.getLogger(__name__)


############

### merging the barcodes part.
def merge_barcodes(A,B):
    B=B.copy()  # start with B's version (this is arbitrary!)
    B[np.isnan(B)]=A[np.isnan(B)] # but if B is ignorant, use info from A
    return B

def codebook_deduplication_iteration(barcodes, thre):
    '''
    Take a codebook with possibly repeated barcodes (or nearly
    repeated, i.e. two barcodes within thresh) and get a codebook
    which is just a little bit cleaner.

    Out
    - improved: boolean, whether we did anything useful
    - barcodes: array, new cleaner codebook
    '''
    R,C,J=barcodes.shape
    barcodes=barcodes.reshape((-1,barcodes.shape[-1]))

    # for each pair of barcodes, find out which frames they disagree on
    differences = np.abs(barcodes[:,None,:] - barcodes [:,:,None]).astype(float) # N x J x J

    # if one of the barcodes says it doesn't know about one of the frames
    # then we say that barcode doesnt disagree about that fram
    differences[np.isnan(differences)]=0

    # compute the total number of disagreements for each pair of barcodes
    differences=np.sum(differences,axis=0) # J x J

    # not interested in barcodes similarity to themselves!
    differences[np.r_[0:J],np.r_[0:J]]=np.inf

    # find, for each barcode, the barcode which is closest to it
    closest_barcodes=np.min(differences,axis=1) # length J, stores hamming distance to the closest barcodes.

    # find barcodes which are TOO CLOSE to some other barcode
    bad_barcodes = np.where(closest_barcodes <= thre)[0]

    if len(bad_barcodes)==0:
        # no such barcodes!  done!
        return True,barcodes.reshape((R,C,-1))
    else:
        # merge the barcodes which are too close
        j1=bad_barcodes[0]
        j2=np.argmin(differences[j1])
        barcodes = barcodes.T
        mergycode=merge_barcodes(barcodes[j1],barcodes[j2])
        barcodes=list(barcodes)
        barcodes.pop(j2)
        barcodes[j1]=mergycode
        return False,np.array(barcodes).T.reshape((R,C,-1))

def codebook_deduplication(barcodes, thre = 1,onehot=True):
    while True:
        done,barcodes=codebook_deduplication_iteration(barcodes, thre)
        if done:
            return barcodes

def trivial_codebook_deduplication(cb):
    R,C,J=cb.shape
    cb=cb.reshape((R*C,-1)).T

    cb1=[cb[0]]
    for i in range(1,len(cb)):
        newb=sp.spatial.distance.cdist([cb[i]],cb1)
        if (newb>0).all():
            cb1.append(cb[i])
    return np.array(cb1).T.reshape((R,C,-1))

#################


def get_denselearner_reconstruction(X,codebook,blob_radius,blur_level=0,
                n_unused_barcodes=2,bardensr_spot_thresh_multiplier=1.0,niter=120,lam=.01):
    # add extra barcodes
    R,C,J=codebook.shape
    unused_barcodes=misc.convert_codebook_to_onehot_form(npr.randint(0,C,size=(n_unused_barcodes,R)))
    estimated_plus_phony = np.concatenate([codebook,unused_barcodes],axis=-1)

    # run bardensr
    estimated_plus_phony[np.isnan(estimated_plus_phony)]=0.0
    bdresult=singlefov.denselearner.build_density(X,estimated_plus_phony,use_tqdm_notebook=True,
                                                          niter=niter,lam=lam,blur_level=blur_level)

    # get spots we consider "good enough", use this to form a mask
    thresh=bdresult.density[:,:,:,-n_unused_barcodes:].max()*bardensr_spot_thresh_multiplier
    mask=bdresult.density>thresh

    # dilate the mask
    newmask=[]
    for j in range(mask.shape[-1]):
        m=ellipsoid_dilation(mask[:,:,:,j],(0,blob_radius,blob_radius))
        newmask.append(m)
    newmask=np.stack(newmask,axis=-1)

    # get the residual
    RD=bdresult.reconstruction_density.copy()
    RD[~newmask]=0
    recon = np.einsum('xyzj,rcj->rcxyz',RD,bdresult.reconstruction_codebook)

    return recon

def blurry_3d_nmf(X,blur):
    raise NotImplementedError

def trivial_rank1_nmf(X):
    '''
    Input: X, (N x M)
    Output:
    - U, an N-vector
    - V, an M-vector
    '''
    U,e,V=sp.sparse.linalg.svds(X,1)

    if np.sum(U)<0:
        U=-U
        V=-V

    U[U<0]=0
    V[V<0]=0

    return U.ravel(),V.ravel()


def find_one_barcode(img,nmf_method='svd',
                    nmf_options=None,
                    proportion_good_rounds_required=.8,
                    thre_onehot=2,
                    r2_thre=.8,
                    global_r2=True):
    '''
    Input:
    - img: a patch (R x C X M0 x M1 x M2)
    - nmf_method: svd or nonnegopt
    - nmf_options: any options that should get passed to the nmf method
    - proportion_good_rounds_required: scalar

    Output:
    - loooksgood: boolean, whether we could detect a rolony here
    - barcode: (R x C), binary array
    '''

    R,C,M0,M1,M2=img.shape
    N=R*C

    # get blob radius info, based on patch size
    blob_size=np.array(img.shape[2:])
    assert (blob_size%2==1).all() # otherwise the patch has no center!
    blob_radius=(blob_size-1)//2

    # further refine the patch using a mask
    mask = skimage.draw.ellipsoid(*(blob_radius+1))[2:-2,2:-2,2:-2]
    img[:,:,~mask]=0.0

    # run svd on whole image
    if nmf_method=='svd':
        Vh,U=trivial_rank1_nmf(img.reshape((N,-1)))
        U=U.reshape(img.shape[2:])
    elif nmf_method=='blurry':
        Vh,U  = blurry_3d_nmf(img.reshape((N,-1)),BLR=blob_radius//2)
    else:
        raise Exception("NYI")

    # find channels that look promising
    # Note: in some rounds there may not be
    # sufficient evidence about which channel ought to
    # be lighting up in that round.
    # We leave those blank...
    good_channels=np.argmax(Vh.reshape(R, C), axis = 1)
    newbarcode = np.zeros((R, C), dtype = 'bool')
    v = Vh.reshape((R, C))
    for r, c in enumerate(good_channels):
        if v[r, c] > thre_onehot*np.mean(v[r]):
            newbarcode[r, c] = True
    n_good_rounds=np.sum(newbarcode)

    # ------> if there are too many trivial rounds, abort!
    if n_good_rounds < int(R*proportion_good_rounds_required):
        return False,None

    # restrict to frames that involve the frames we believe
    # are relevant to this barcode
    img_just_good_channels = img[newbarcode]

    # rerun svd on just the good frames
    if nmf_method=='svd':
        Vh,U=trivial_rank1_nmf(img_just_good_channels.reshape((n_good_rounds,-1)))
        U=U.reshape(img.shape[2:])
    elif nmf_method=='blurry':
        Vh,U  = blurry_3d_nmf(img_just_good_channels,BLR=blob_radius//2)
    else:
        raise Exception("NYI")

    # now comes the hard part.  how do we decide if
    # this new barcode we found is real?  cosine similarity
    # between original image and the reconstruction from
    # our rank-1 model (note the reconstruction is ZERO
    # for all frames which are irrelevant to the barcode)

    # compute dot product
    dot=np.sum(img_just_good_channels * Vh[:,None,None,None] * U[None,:,:,:])

    # compute normalized dot product
    if global_r2:
        ndp = dot / np.sqrt(np.sum(Vh**2)*np.sum(U**2)*np.sum(img**2))
    else:
        ndp = dot / np.sqrt(np.sum(Vh**2)*np.sum(U**2)*np.sum(img_just_good_channels**2))

    # ----> if if doesn't fit, abort!  abandon this blob!
    if ndp < r2_thre:
        return False,None

    # convert rows with all zeros to nan
    newbarcode=newbarcode.astype(np.float)
    for r in range(R):
        if np.sum(newbarcode[r])==0:
            newbarcode[r]=np.nan

    # ----> success!
    return True, newbarcode


def seek_barcodes(X,
                               blob_radius,
                               thre_onehot = 2,
                               proportion_good_rounds_required = 1.0,
                               r2_thre = 0.8,
                               blob_thre = 0.2,
                               nmf_method = 'svd',
                               nmf_options=None,
                               global_r2=True,
                              ):
    '''
    find potential barcodes from X.  we run blob detection, and then for each blob we
    - grab a patch around that blob
    - try to match it with a rank-1 model
    - if we succeed, we call that a success and add it to our barcode list

    Input:
        X: must be (R, C, M0, M1, M2)
        blob_radius: a tuple of 3 integers indicating the expected radius of a blob along the M0,M1,M2 axes
            (use 0 if a blob may be only one voxel big, e.g. if there is only one slice of Z)
        thre_onehot: a round is considered only if one channel is thre_onehot times greater than the mean
        proportion_good_rounds_required: for a spot to be "good", at least this proportion
            of the rounds must be considered
        r2_thre: for a spot to be "good", at least this proportion of the variance
            must be explained by the rank-1 model
        blob_thre: for a spot to be "good", it must be at least this bright
            (relative to overall image max)

    Output: Information about all valid spots we found.  Let S denote number of spots found.
        barcode_list: numpy array of (R, C, S) -- found barcodes for S spots.
        blob_out: numpy array of (S, 3) -- coordinates for S spots.
    '''
    R, C, M0,M1, M2 = X.shape
    N = R*C

    # we will be considering patches of this size
    blob_radius=np.array(blob_radius,dtype=np.int)
    blob_patch_npix=np.prod(blob_radius*2+1)

    barcode_list = [] # barcodes discovered!
    blob_out = []     # rolonies discovered!

    # find some blobs
    blobs = skimage.feature.peak_local_max(np.max(X, axis = (0, 1)),
                                           threshold_abs = blob_thre*X.max(),
                                           exclude_border=False,
                                           min_distance = 2
                                          )

    # get rid of the ones on the edge.  they are confusing
    good=True
    for i in range(3):
        good=good&(blobs[:,i]>=blob_radius[i])
        good=good&(blobs[:,i]<=X.shape[i+2]-blob_radius[i]-1)
    n_original_blobs=len(blobs)
    n_non_edge=np.sum(good)
    blobs=blobs[good]


    # go through each blob and check its awesomeness
    for i, loc in enumerate(blobs):

        # get a local patch around the blob
        xslice = (slice(0,None),)*2+tuple([slice(x-rd,x+rd+1) for (x,rd) in zip(loc,blob_radius)])
        img = X[xslice].copy() # R x C x [blob_size]

        # see if we can find a barcode in that blob
        good,newbarcode=find_one_barcode(img,nmf_method=nmf_method,nmf_options=nmf_options,
                        r2_thre=r2_thre,thre_onehot=thre_onehot,
                        proportion_good_rounds_required=proportion_good_rounds_required,
                        global_r2=global_r2)

        # if its good, save it
        if good:
            barcode_list.append(newbarcode)
            blob_out.append(loc)

    logger.debug(f"blobs detected ({n_original_blobs} total --> {n_non_edge} non-edge --> {len(blob_out)} good)")

    if len(barcode_list)>0:
        return np.transpose(np.array(barcode_list),[1,2,0]), np.array(blob_out)
    else:
        return np.zeros((R,C,0),dtype=np.bool),np.zeros((0,3))

def ellipsoid_dilation(mask, blob_radius):
    blob_radius=np.array(blob_radius)
    dilation_structure = skimage.draw.ellipsoid(*(blob_radius+1))[2:-2,2:-2,2:-2]
    return skimage.morphology.dilation(mask, dilation_structure)
