def test_distributed_estimation():
    import bardensr
    import numpy as np
    import tensorflow as tf
    import numpy.random as npr
    # make data
    npr.seed(0)
    data=np.zeros((5,20,30,40))
    pts=np.array([npr.randint(0,x,size=5) for x in data.shape[1:]]).T
    for f in range(5):
        data[(f,)+tuple(pts[f])]=1

    # get pairwise correlations
    D=np.zeros((3,5,5))
    for f1 in range(5):
        for f2 in range(5):
            D[:,f1,f2] = bardensr.registration.pairwise_correlation_registration(data[f1],data[f2])

    # try it out!
    reconciliation=bardensr.registration.distributed_translation_estimator(D).T
    with tf.device('cpu'):
        newd,newt=bardensr.registration.apply_translations(data,reconciliation,'valid')
    assert np.allclose(np.sum(np.prod(newd,axis=0)),1)

def test_dotproducts():
    import bardensr
    import numpy as np
    A=np.zeros((20,30,40))
    B=np.zeros((20,30,40))
    A[2,9,4]=1
    B[7,3,11]=1
    V,offset=bardensr.registration.dotproducts_at_translations(A,B,demean=True,cutin=[3,2,4])
    assert np.allclose(bardensr.registration.pairwise_correlation_registration(A,B),[-5,6,-7])

def test_find_translations_using_model():
    import bardensr
    import numpy as np
    import numpy.random as npr
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
    for i in range(20):
        m0=npr.randint(0,50)
        m1=npr.randint(0,50)
        j=npr.randint(0,n_codes)
        density[m0,m1,j]=npr.rand()+1
    density=sp.ndimage.gaussian_filter(density,(1,1,0))

    # make image
    image=np.einsum('xyj,nj->nxy',density,codebook)

    # get loss
    oloss=bardensr.registration.lowrankregistration_tf._calc_scaled_loss(
        image,codebook,np.ones((n_frames,2))*(-5),np.r_[60,60])
    assert oloss<.05

    # pick translation
    t=-npr.rand(n_frames,2)*2
    sz=np.r_[60,60]

    # translate image
    image2=bardensr.registration.translations_tf.floating_slices(
        image,t,sz,'hermite')

    # make sure the right answer gives pretty good loss
    fixloss=bardensr.registration.lowrankregistration_tf._calc_scaled_loss(image2,codebook,-t-10,np.r_[70,70])
    assert fixloss<oloss+.05

    badloss=bardensr.registration.lowrankregistration_tf._calc_scaled_loss(image2,codebook,t*0-10,np.r_[70,70])


    t2=bardensr.registration.find_translations_using_model(image2[:,:,None,:],codebook,niter=100)

    t2_normalized=t2-np.mean(t2,axis=0,keepdims=True)
    t1_normalized=np.c_[t[:,0],np.zeros(n_frames),t[:,1]]
    t1_normalized=t1_normalized-np.mean(t1_normalized,axis=0,keepdims=True)

    assert np.mean(np.abs(t1_normalized+t2_normalized))<.05

    image3,_=bardensr.registration.apply_translations(image2[:,:,None,:],t2)
    otherloss=bardensr.registration.lowrankregistration_tf._calc_scaled_loss(np.squeeze(image3),codebook,t*0-10,np.r_[70,70])
    assert otherloss<.1