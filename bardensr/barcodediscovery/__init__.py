import skimage
import skimage.draw
import skimage.feature
import numpy as np
import scipy as sp
import scipy.sparse.linalg
import numpy.random as npr

from .. import misc
from .. import singlefov
from . import purepixel

import logging
logger=logging.getLogger(__name__)


############

### merging the barcodes part.
def merge_barcodes(A,B):
    B=B.copy()  # start with B's version (this is arbitrary!)
    B[np.isnan(B)]=A[np.isnan(B)] # but if B is ignorant, use info from A
    return B

def nan_robust_hamming(A,B):
    '''
    Input:
    - A (M x N1 )
    - B (M x N2 )

    Output
    - diffs (N1 x N2)

    diffs[i,j] = #{k: both A[i,k],B[i,k] are non-nan and A[i,k]!=B[i,k}
    '''

    differences = np.abs(A[:,None,:] - B[:,:,None]).astype(float) # M x N1 x N2

    # if one of the barcodes says it doesn't know about one of the frames
    # then we say that barcode doesnt disagree about that fram
    differences[np.isnan(differences)]=0

    # compute the total number of disagreements for each pair of barcodes
    differences=np.sum(differences,axis=0) # N1 x N2

    return differences

def codebook_deduplication_iteration(barcodes, thre,differences=None):
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

    if differences is None:
        # calc hamming (but ignore nans!)
        differences = nan_robust_hamming(barcodes,barcodes)
        # not interested in barcodes similarity to themselves!
        differences[np.r_[0:J],np.r_[0:J]]=np.inf

    # find, for each barcode, the barcode which is closest to it
    closest_barcodes=np.min(differences,axis=1) # length J, stores hamming distance to the closest barcodes.

    # find barcodes which are TOO CLOSE to some other barcode
    bad_barcodes = np.where(closest_barcodes <= thre)[0]

    if len(bad_barcodes)==0:
        # no such barcodes!  done!
        return True,barcodes.reshape((R,C,-1)),differences
    else:
        # merge the barcodes which are too close
        j1=bad_barcodes[0]
        j2=np.argmin(differences[j1])
        assert j1<j2
        mergycode=merge_barcodes(barcodes[:,j1],barcodes[:,j2])

        # get a new barcode list by removing j2 and replacing j1
        goodcodes=np.r_[0:j2-1,j2:barcodes.shape[1]]
        barcodes=barcodes[:,goodcodes]
        barcodes[:,j1]=mergycode

        # get new diffs by removing j2 and replacing j1
        differences=differences[goodcodes]
        differences=differences[:,goodcodes]
        differences[j1]=nan_robust_hamming(barcodes[:,[j1]],barcodes).ravel()
        differences[:,j1]=differences[j1].ravel()
        differences[j1,j1]=np.inf

        # done!
        return False,np.array(barcodes).reshape((R,C,-1)),differences

def codebook_deduplication(barcodes, thre = 1,onehot=True,use_tqdm_notebook=False):
    diffs=None
    if use_tqdm_notebook:
        import tqdm.notebook
        with tqdm.notebook.tqdm(leave=False) as t:
            while True:
                t.update(1)
                done,barcodes,diffs=codebook_deduplication_iteration(barcodes, thre,diffs)
                if done:
                    return barcodes
    else:
        while True:
            done,barcodes,diffs=codebook_deduplication_iteration(barcodes, thre,diffs)
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


def blurry_3d_nmf_lossfunc(data, G, F,blur):
    '''
    data -- (...,N,M0,M1,M2)
    G -- (...,N)
    F -- (...,M0,M1,M2)
    '''
    from .. import kernels
    import tensorflow as tf
    F=kernels.gaussian_filter_3d(F,blur)
    reconstruction = tf.einsum('...xyz,...n->...nxyz',F,G)
    return tf.reduce_sum((reconstruction-data)**2)

from .. import nonnegopt
blurry_3d_nmf_improve_F = nonnegopt.cnuwb_speedify(blurry_3d_nmf_lossfunc,'F', 0)
blurry_3d_nmf_improve_G = nonnegopt.cnuwb_speedify(blurry_3d_nmf_lossfunc,'G', 0)

def blurry_3d_nmf(data,blur,n_iter=5,track_losses=False):
    '''
    Input:
    - data (batch,N,M0,M1,M2)
    - blur, a 3-tuple of ints
    - n_iter, number of iterations

    Output:
    - V - temporal vector (batch,N,)
    - U - spatial vecotr (batch,M0,M1,M2,)
    - losses over time
    '''
    from .. import nonnegopt
    import tensorflow as tf
    assert(data.min() >=0)

    blur=tuple([int(x) for x in blur])
    assert len(blur)==3

    loss_list = []

    batch,N,M0,M1,M2=data.shape

    initial_guess_for_G,initial_guess_for_F=trivial_rank1_nmf(data.reshape((batch,N,-1)))
    initial_guess_for_F=initial_guess_for_F.reshape((batch,M0,M1,M2))
    initial_guess_for_G = tf.convert_to_tensor(initial_guess_for_G)
    initial_guess_for_F = tf.convert_to_tensor(initial_guess_for_F)

    data =tf.convert_to_tensor(data)
    state=dict(data = data, F=initial_guess_for_F, G =initial_guess_for_G,blur=blur)

    if track_losses:
        loss_list.append(blurry_3d_nmf_lossfunc(**state).numpy())
    for i in range(n_iter):
        state['F'] = blurry_3d_nmf_improve_F(**state)
        if track_losses:
            loss_list.append(blurry_3d_nmf_lossfunc(**state).numpy())
        state['G'] = blurry_3d_nmf_improve_G(**state)
        if track_losses:
            loss_list.append(blurry_3d_nmf_lossfunc(**state).numpy())

    out_F = state['F'].numpy()
    out_G = state['G'].numpy()
    return out_G, out_F,np.array(loss_list)

def trivial_rank1_nmf(X):
    '''
    Input: X, (batch x N x M)
    Output:
    - U, an (batch x N)
    - V, an (batch x M)
    '''

    batch,N,M=X.shape

    Us=[]
    Vs=[]
    for i in range(batch):
        U,e,V=sp.sparse.linalg.svds(X[i],1)
        if np.sum(U)<0:
            U=-U
            V=-V
        U[U<0]=0
        V[V<0]=0
        Us.append(U.ravel()*e)
        Vs.append(V.ravel())
    return np.stack(Us),np.stack(Vs)

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
        Vh,U=trivial_rank1_nmf(img.reshape((N,-1))[None])
        U=U[0]
        Vh=Vh[0]
        U=U.reshape(img.shape[2:])
    elif nmf_method=='nonnegopt':
        Vh,U  = blurry_3d_nmf(img.reshape((N,M0,M1,M2))[None],blur=blob_radius//2)[:2]
        Vh=Vh[0]
        U=U[0]
    else:
        raise NotImplementedError

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
        return False,None,'badrounds'

    # restrict to frames that involve the frames we believe
    # are relevant to this barcode
    img_just_good_channels = img[newbarcode]

    # rerun svd on just the good frames
    if nmf_method=='svd':
        Vh,U=trivial_rank1_nmf(img_just_good_channels.reshape((n_good_rounds,-1))[None])
        Vh=Vh[0]
        U=U[0]
        U=U.reshape(img.shape[2:])
    elif nmf_method=='nonnegopt':
        Vh,U  = blurry_3d_nmf(img_just_good_channels[None],blob_radius//2)[:2]
        Vh=Vh[0]
        U=U[0]
    else:
        raise NotImplementedError

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
        return False,None,'badr2'

    # convert rows with all zeros to nan
    newbarcode=newbarcode.astype(np.float)
    for r in range(R):
        if np.sum(newbarcode[r])==0:
            newbarcode[r]=np.nan

    # ----> success!
    return True, newbarcode,'success'


def seek_barcodes_vectorized(X,
                               blob_radius,
                               thre_onehot = 2,
                               proportion_good_rounds_required = 1.0,
                               r2_thre = 0.8,
                               blob_thre = 0.2,
                               nmf_method = 'svd',
                               nmf_options=None,
                               global_r2=True,
                               blur_radius=None
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

    # create an ellipsoidal mask
    mask = skimage.draw.ellipsoid(*(blob_radius+1))[2:-2,2:-2,2:-2]

    # collect all the blobs
    patches=[]
    for i,loc in enumerate(blobs):
        xslice=(slice(0,None),)*2+tuple([slice(x-rd,x+rd+1) for (x,rd) in zip(loc,blob_radius)])
        patches.append(X[xslice])
    patches=np.array(patches)
    n_patches=len(patches)

    # mask all the blobs
    patches[:,:,:,~mask]=0

    # put all the frames together for the moment
    patches=patches.reshape((n_patches,N,)+patches.shape[-3:])

    # run nmf on all the blobs
    if nmf_method=='svd':
        V,U=trivial_rank1_nmf(patches.reshape((len(patches),N,-1)))
        U=U.reshape(U.shape[:1]+(patches.shape[-3:]))
    elif nmf_method=='nonnegopt':
        V,U,losses=blurry_3d_nmf(patches.reshape((len(patches),N)+patches.shape[-3:]),blur_radius)
    else:
        raise ValueError(f"what is {nmf_method}?")

    # split the frames out again
    V=V.reshape((n_patches,R,C))

    # find channels that look promising
    good_channels=np.argmax(V, axis = 2) # n_patches x R

    # create putative barcodes based on those channels
    codebook = misc.convert_codebook_to_onehot_form(good_channels) # R x C x J

    # in some rounds there may not be
    # sufficient evidence about which channel ought to
    # be lighting up in that round.
    # create a second codebook that leaves those blank..
    top_brightness=np.max(V,axis=2)
    mean_brightness=np.mean(V,axis=2)
    happy=top_brightness> thre_onehot*mean_brightness
    codebook2=codebook.copy()
    for j,r in zip(*np.where(~happy)):
        codebook2[r,:,j]=False

    # how many good rounds are there for each patch?
    n_good_rounds=np.sum(codebook2,axis=(0,1))

    # get rid of the bad guys
    bad=n_good_rounds < int(R*proportion_good_rounds_required)
    n_good_rounds=n_good_rounds[~bad]
    codebook=codebook[:,:,~bad]
    codebook2=codebook2[:,:,~bad]
    patches=patches[~bad]

    logger.debug(f"num_bad_rounds={np.sum(bad)}")

    # now restrict our attention to the relevant channels
    patches_limited=np.array([
            patches[j][codebook[:,:,j].ravel()]
        for j in range(len(patches))
    ]) # batch x R x M0 x M1 x M2

    # rerun svd
    if nmf_method=='svd':
        V,U=trivial_rank1_nmf(patches_limited.reshape((len(patches_limited),R,-1)))
        U=U.reshape((-1,)+patches_limited.shape[-3:])
    elif nmf_method=='nonnegopt':
        V,U,losses=blurry_3d_nmf(patches_limited,blur_radius)
    else:
        raise ValueError(f"what is {nmf_method}?")

    # compute dot product for each patch
    dot=np.sum(patches_limited * V[:,:,None,None,None] * U[:,None,:,:,:],axis=(1,2,3,4))

    # compute normalized dot product
    reconsq=np.sum(V**2,axis=1) * np.sum(U**2,axis=(1,2,3))
    if global_r2:
        datasq=np.sum(patches**2,axis=(1,2,3,4))
    else:
        datasq=np.sum(patches_limited**2,axis=(1,2,3,4))
    ndp=dot/np.sqrt(reconsq*datasq)

    # get rid of bad guys again
    bad=ndp < r2_thre
    n_good_rounds=n_good_rounds[~bad]
    codebook=codebook[:,:,~bad]
    codebook2=codebook2[:,:,~bad]
    patches=patches[~bad]

    # turn empty rows into nans
    bad_rounds = np.sum(codebook2,axis=1)==0 # R x J
    codebook2=codebook2.astype(np.float)
    for r,j in zip(*np.where(bad_rounds)):
        cchoice=np.where(codebook[r,:,j])[0]
        codebook2[r,cchoice,j]=np.nan

    # done
    return codebook2

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
    messages=[]
    for i, loc in enumerate(blobs):

        # get a local patch around the blob
        xslice = (slice(0,None),)*2+tuple([slice(x-rd,x+rd+1) for (x,rd) in zip(loc,blob_radius)])
        img = X[xslice].copy() # R x C x [blob_size]

        # see if we can find a barcode in that blob
        good,newbarcode,msg=find_one_barcode(img,nmf_method=nmf_method,nmf_options=nmf_options,
                        r2_thre=r2_thre,thre_onehot=thre_onehot,
                        proportion_good_rounds_required=proportion_good_rounds_required,
                        global_r2=global_r2)
        messages.append(msg)

        # if its good, save it
        if good:
            barcode_list.append(newbarcode)
            blob_out.append(loc)

    logger.debug(str(np.unique(messages,return_counts=True)))

    if len(barcode_list)>0:
        return np.transpose(np.array(barcode_list),[1,2,0])
    else:
        return np.zeros((R,C,0),dtype=np.bool)

def ellipsoid_dilation(mask, blob_radius):
    blob_radius=np.array(blob_radius)
    dilation_structure = skimage.draw.ellipsoid(*(blob_radius+1))[2:-2,2:-2,2:-2]
    return skimage.morphology.dilation(mask, dilation_structure)
