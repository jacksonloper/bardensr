import numpy as np
from .. import misc
import numpy.random as npr
import pandas as pd
import scipy as sp
from .. import meshes

import logging

logger=logging.getLogger(__name__)

def simulate_codebook(R,C,J):
    A=npr.randint(0,C,size=(J,R))
    return misc.convert_codebook_to_onehot_form(A)

def mess_up_barcode(barcode,signal_range,per_frame_signal_range,dropout_probability):
    R,C=barcode.shape
    barcode=barcode.copy().astype(np.float64)
    barcode[npr.rand()<.5]=0 # dropout
    barcode=barcode*(npr.rand()*(signal_range[1]-signal_range[0]) + signal_range[0])
    barcode = barcode *(npr.rand(R,C)*(signal_range[1]-signal_range[0]) + signal_range[0])
    return barcode

def prepare_meshes_for_benchmark(meshlist,pitch,poisson_rate,num_workers=1,use_tqdm_notebook=False,
                                                    maxout_count=np.inf):
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

    translation=np.min([np.min(x.vertices,axis=0) for x in meshlist],axis=0)
    new_meshes=[trimesh.Trimesh((x.vertices-translation[None,:])/pitch,x.faces) for x in meshlist]

    voxel_lists=meshes.voxelize_meshlist_interiors(new_meshes,pitch=1.0,
                            num_workers=num_workers,use_tqdm_notebook=use_tqdm_notebook)

    locations=[x.astype(np.int) for x in voxel_lists]
    locations=[np.c_[x,np.full(len(x),i)] for i,x in enumerate(locations)]
    locations=np.concatenate(locations,axis=0)
    GT_voxels=pd.DataFrame(dict(
        m0=locations[:,0],
        m1=locations[:,1],
        m2=locations[:,2],
        j=locations[:,3],
    ))

    counts=npr.poisson(poisson_rate*(pitch*pitch*pitch),size=len(locations))
    if maxout_count is not np.inf:
        counts[counts>maxout_count]=maxout_count
    locations2=np.repeat(locations,counts,axis=0)
    rolonies=pd.DataFrame(dict(
        m0=locations2[:,0],
        m1=locations2[:,1],
        m2=locations2[:,2],
        j=locations2[:,3],
    ))

    return new_meshes,GT_voxels,rolonies,translation

def simulate_imagestack(rolonies,codebook,
                dropout_probability=0.0,
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
        bc = mess_up_barcode(codebook[:,:,j],signal_range,per_frame_signal_range,dropout_probability)
        X[:,:,m0,m1,m2]+=bc

    if blursz is not None:
        blurs=(0,0)+tuple(blursz)
        logger.info('blurring' + str(blurs))
        X=sp.ndimage.gaussian_filter(X,blurs)
    X=X+np.random.randn(*X.shape)*speckle_noise
    X=np.clip(X,0,None)
    return X
