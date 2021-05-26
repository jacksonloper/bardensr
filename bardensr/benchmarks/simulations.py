import numpy as np
from .. import misc
import numpy.random as npr
import pandas as pd
import scipy as sp
from . import locsdf

import logging

logger=logging.getLogger(__name__)

def simulate_codebook(R,C,J):
    A=npr.randint(0,C,size=(J,R))
    return misc.convert_codebook_to_onehot_form(A)

def mess_up_barcode(barcode,signal_range, per_frame_signal_range,
                    dropout_probability, num_dropout_r = 1, dropout_intensity = 0.1):
    R,C=barcode.shape
    barcode=barcode.copy().astype(np.float64)
    barcode=barcode*(npr.rand()*(signal_range[1]-signal_range[0]) + signal_range[0])
    barcode = barcode *(npr.rand(R,C)*(per_frame_signal_range[1]-per_frame_signal_range[0]) + per_frame_signal_range[0])
    if npr.rand() < dropout_probability:  # Dropout this spot!
        DO_r = npr.choice(R, size = num_dropout_r, replace = False)
        barcode[DO_r] *= dropout_intensity
    return barcode

def prepare_meshes_for_benchmark(meshlist,pitch,poisson_rate,num_workers=1,use_tqdm_notebook=False):
    '''
    Moves meshes into a new coordinate system, voxelizes them
    in that coordinate system, and generates rolonies inside
    the voxelizations

    Input:
    - meshlist, a list of watertight trimesh.Trimesh objects
    - pitch, length of voxel

    Output:
    - new_meshes, a new list of watertight trimesh.Trimesh objects, in a new reference frame
    - GT_voxels, a dataframe indicating positions of all voxels inside the meshes
    - rolonies, a dataframe indicating rolonies inside the meshes, sampled with poisrate in the original units
    - translation, a vector indicating how original meshes were translated

    Meshes are moved into a new coordinate system, such that

        new_meshes[5].vertices = (meshlist[5].vertices - translation)/pitch

    GT_voxels then indicate positions which are contained within each of these
    transformes meshes, i.e. if we have

    GT_voxels.iloc[3]:
        m0 = 3
        m1 = 10
        m2 = 12
        j  = 15

    That signifies that the point [3,10,12] is (approximately) contained inside new_meshes[15]
    '''

    import trimesh
    from .. import meshes

    #########################################
    # voxelize!

    # translate meshes to a reference frame where one voxel = one unit
    translation=np.min([np.min(x.vertices,axis=0) for x in meshlist],axis=0)
    new_meshes=[trimesh.Trimesh((x.vertices-translation[None,:])/pitch,x.faces) for x in meshlist]

    # do the voxelation
    voxel_lists=meshes.voxelize_meshlist_interiors(new_meshes,pitch=1.0,
                            num_workers=num_workers,use_tqdm_notebook=use_tqdm_notebook)

    # record the locations as a single big dataframe
    locations=[x.astype(np.int) for x in voxel_lists]
    locations=[np.c_[x,np.full(len(x),i)] for i,x in enumerate(locations)]
    locations=np.concatenate(locations,axis=0)
    GT_voxels=pd.DataFrame(dict(
        m0=locations[:,0],
        m1=locations[:,1],
        m2=locations[:,2],
        j=locations[:,3],
    ))

    #########################################
    # sample transcripts inside the voxels
    # this is tricky because in some cases multiple
    # neurons inhabit the same voxel.  in this case
    # we assume each neuron occupies an equal proportion

    # get a list of all locations in GT voxels
    # note that locs are not unique, i.e. we may have locs[3]==locs[4]
    # as long as js[3]!=js[4].
    locs, js = locsdf.df_to_locs_and_j(GT_voxels)

    # for each (nonunique!) location
    # compute how many total neurons inhabit that voxel
    locs_unique, idx, cts = np.unique(
        locs, axis=0, return_inverse=True, return_counts=True)
    inhabitants = cts[idx] # for each entry in GT_voxels

    # for each (nonunique!) location,
    # sample poisson, with rate inversely proportional
    # to number of inhabiting neurons
    counts = npr.poisson(poisson_rate*(pitch*pitch*pitch)/inhabitants)

    # create final tally
    locs_repeated=np.repeat(locs,counts,axis=0)
    js_repeated=np.repeat(js,counts,axis=0)
    rolonies=pd.DataFrame(dict(
        m0=locs_repeated[:,0],
        m1=locs_repeated[:,1],
        m2=locs_repeated[:,2],
        j=js_repeated,
    ))

    #########################################
    return new_meshes,GT_voxels,rolonies

def simulate_imagestack(rolonies,codebook,
                dropout_probability=0.0,
                num_dropout_r = 1,
                dropout_intensity = 0.1,
                speckle_noise=.01,
                signal_range=(10,15),
                per_frame_signal_range=(1.0,1.0),
                blursz=(2,2,2),
                use_tqdm_notebook=True,
    ):
    '''
    Input
    - rolonies, a dataframe with voxel positions m0,m1,m2 and j (index)
    - codebook, a codebook!
    - [optional] dropout_probability -- the proportion of the spots that should be dropped out
    - [optional] num_dropout_r -- the number of rounds that drops out
    - [optional] dropout_intensity -- if the fram signal is dropped out, the original intensity compared to original
    - [optional] speckle_noise -- how much iid gaussian noise to add to the data
    - [optional] signal_range -- how much rolonies vary in total intensity
    - [optional] per_frame_signal_range -- how much rolonies will vary in intensity between rounds/channels
    - [optional] blursz -- how much blur in each dimension

    Note that voxel positions should index into a voxel array, i.e.
    they should be all be nonnegative integers.
    '''

    R,C,J=codebook.shape

    shape=(rolonies['m0'].max()+1,rolonies['m1'].max()+1,rolonies['m2'].max()+1)
    X=np.zeros((R, C)+shape)

    logger.info('inserting points')
    for i in misc.maybe_trange(len(rolonies),use_tqdm_notebook):
        val=rolonies.iloc[i]
        m0=val.m0
        m1=val.m1
        m2=val.m2
        j=val.j
        bc = mess_up_barcode(codebook[:,:,j],signal_range,per_frame_signal_range,
                             dropout_probability,num_dropout_r,dropout_intensity)
        X[:,:,m0,m1,m2]+=bc

    if blursz is not None:
        blurs=(0,0)+tuple(blursz)
        logger.info('blurring' + str(blurs))
        X=sp.ndimage.gaussian_filter(X,blurs)
    X=X+np.random.randn(*X.shape)*speckle_noise
    X=np.clip(X,0,None)
    return X

def simulate_imagestack_bcs(rolonies,codebook,
                dropout_probability=0.0,
                num_dropout_r = 1,
                dropout_intensity = 0.1,
                speckle_noise=.01,
                signal_range=(10,15),
                per_frame_signal_range=(1.0,1.0),
                blursz=(2,2,2),
                use_tqdm_notebook=True,
    ):
    '''
    Input
    - rolonies, a dataframe with voxel positions m0,m1,m2 and j (index)
    - codebook, a codebook!
    - [optional] dropout_probability -- the proportion of the spots that should be dropped out
    - [optional] num_dropout_r -- the number of rounds that drops out
    - [optional] dropout_intensity -- if the fram signal is dropped out, the original intensity compared to original
    - [optional] speckle_noise -- how much iid gaussian noise to add to the data
    - [optional] signal_range -- how much rolonies vary in total intensity
    - [optional] per_frame_signal_range -- how much rolonies will vary in intensity between rounds/channels
    - [optional] blursz -- how much blur in each dimension

    Note that voxel positions should index into a voxel array, i.e.
    they should be all be nonnegative integers.
    '''

    R,C,J=codebook.shape

    shape=(rolonies['m0'].max()+1,rolonies['m1'].max()+1,rolonies['m2'].max()+1)

    bcs=[]

    logger.info('inserting points')
    for i in misc.maybe_trange(len(rolonies),use_tqdm_notebook):
        j=val.j
        bc = mess_up_barcode(codebook[:,:,j],signal_range,per_frame_signal_range,
                             dropout_probability,num_dropout_r,dropout_intensity)
        bcs.append(bc)

    return bcs

def bcs_to_imagestack(rolonies,bcs,js_to_skip=None):

    if js_to_skip is None:
        js_to_skip={}

    shape=(rolonies['m0'].max()+1,rolonies['m1'].max()+1,rolonies['m2'].max()+1)

    logger.info('inserting points')
    for i in misc.maybe_trange(len(rolonies),use_tqdm_notebook):
        val=rolonies.iloc[i]
        m0=val.m0
        m1=val.m1
        m2=val.m2
        j=val.j
        if j in js_to_skip:
            pass
        else:
            X[:,:,m0,m1,m2]+=bcs[i]

    if blursz is not None:
        blurs=(0,0)+tuple(blursz)
        logger.info('blurring' + str(blurs))
        X=sp.ndimage.gaussian_filter(X,blurs)
    X=X+np.random.randn(*X.shape)*speckle_noise
    X=np.clip(X,0,None)
    return X

'''
    # step 1, make imagestack
    original_imagestack=np.sum(X,axis=0)

    # step 2, find some barcodes
    barcodes = find_my_happy_barcodes(original_imagestack)

    # step 3A, PERFECTLY SUBTRACT OUT those barcodes, using oracle knowledge
    barcodes_that_I_have_already_found_boolean_mask[j]=(true if I have found that barcode)
    perfect_residual=np.sum(X[~barcodes_that_I_have_already_found_boolean_mask],axis=0)

    # step 3B, IMPERFECTLY SUBTRACT OUT, using bardensr
    barcodes_that_I_have_already_found_boolean_mask[j]=(true if I have found that barcode)
    imperfect_residual=bardensr_reconstruction(original_imagestack,barcodes)

    # step 4 -- compare our ability to find new barcodes in the two
    # different residuals
    extra_barcodesA = find_my_happy_barcodes(perfect_residual)
    extra_barcodesB = find_my_happy_barcodes(imperfect_residual)

    return X
'''
