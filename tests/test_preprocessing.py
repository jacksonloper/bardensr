

def test_bgsubtrac():
    import bardensr
    import numpy as np
    import numpy.random as npr

    A=npr.randn(3,5,6,7)

    B=bardensr.preprocessing.background_subtraction(A,[2,3,4])

    assert A.shape==B.shape

def test_mnmx():
    import bardensr
    import numpy as np
    import numpy.random as npr

    A=npr.randn(3,5,6,7)

    B=bardensr.preprocessing.minmax(A)
    assert B.min()==0
    assert B.max()==1

    B=bardensr.preprocessing.preprocessing_tf.minmax(A,[1,2]).numpy()
    assert B[1,:,:,2].min()==0
    assert B[1,:,:,2].max()==1

def test_gf1():
    import bardensr
    import numpy as np
    import numpy.random as npr

    A=npr.randn(200,30)

    A[:50]=0
    A[-50:]=0

    B=bardensr.preprocessing.preprocessing_tf.gaussian_filter_1d(A,3,0).numpy()

    assert np.allclose(np.sum(A),np.sum(B))

