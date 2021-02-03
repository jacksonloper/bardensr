import numpy as np
from .. import misc
import numpy.random as npr
import pandas as pd
import scipy as sp

import logging

logger=logging.getLogger(__name__)

def simulate_codebook(R,C,J):
    A=npr.randint(0,C,size=(J,R))
    return misc.convert_codebook_to_onehot_form(A)

def convert_rolony_location_list_to_voxel_coordinate_dataframe(locations,pitch):
    locations=[(x//pitch).astype(np.int) for x in locations]
    locations=[np.c_[x,np.full(len(x),i)] for i,x in enumerate(locations)]
    locations=np.concatenate(locations,axis=0)
    assert (np.min(locations,axis=0)[:3]>=0).all(),'negative locations not permitted'
    return pd.DataFrame(dict(
        m0=locations[:,0],
        m1=locations[:,1],
        m2=locations[:,2],
        j=locations[:,3],
    ))

def mess_up_barcode(barcode,signal_range,per_frame_signal_range,dropout_probability):
    R,C=barcode.shape
    barcode=barcode.copy().astype(np.float64)
    barcode[npr.rand()<.5]=0 # dropout
    barcode=barcode*(npr.rand()*(signal_range[1]-signal_range[0]) + signal_range[0])
    barcode = barcode *(npr.rand(R,C)*(signal_range[1]-signal_range[0]) + signal_range[0])
    return barcode

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
