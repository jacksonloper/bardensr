
def test_peakfind():
    import bardensr
    import numpy as np
    import numpy.random as npr

    A=np.zeros((5,6,7,3))

    A[1,2,3,0]=1
    A[2,3,4,1]=1.5
    A[3,4,5,2]=.001

    df=bardensr.spot_calling.find_peaks(A,.5)
    assert len(df)==2
    assert np.sum(df['j']==2)==0

def test_singleshot():
    import bardensr
    import numpy as np
    import numpy.random as npr
    import pandas as pd
    import scipy as sp
    import tensorflow as tf
    import scipy.ndimage
    import matplotlib.pyplot as plt

    n_codes=5
    n_frames=15

    npr.seed(0)

    # make codebook
    codebook=(npr.randn(n_frames,n_codes)>0)*1.0
    codebook=codebook/np.sqrt(np.sum(codebook**2,axis=0,keepdims=True))

    # make density
    density=np.zeros((50,50,n_codes))
    spots=[]
    for i in range(5):
        m0=npr.randint(0,50)
        m1=npr.randint(0,50)
        j=npr.randint(0,n_codes)
        density[m0,m1,j]=npr.rand()+1
        spots.append([m0,0,m1,j])
    spots=pd.DataFrame(data=spots,columns=['m0','m1','m2','j'])

    # make image
    image=np.einsum('xyj,nj->nxy',density,codebook)
    image=sp.ndimage.gaussian_filter(image,(0,1,1))

    # find spots
    V=bardensr.spot_calling.estimate_density_singleshot(image[:,:,None],codebook,noisefloor=.01)
    df=bardensr.spot_calling.find_peaks(V,.8)

    match=bardensr.benchmarks.match_colored_pointclouds(spots,df,5)

    assert match.fn==0
    assert match.fp==0