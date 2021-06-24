

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