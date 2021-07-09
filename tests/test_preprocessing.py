

def test_grad_sfd():
    import bardensr
    import numpy as np
    import numpy.random as npr

    A=npr.randn(5,6,7,8)
    B=bardensr.preprocessing.preprocessing_tf.grad_sfd(A,[1,2])
    assert np.allclose(B[2,:,:,4,0],A[2,2:,1:-1,4]-A[2,:-2,1:-1,4])
    assert np.allclose(B[2,:,:,4,1],A[2,1:-1,2:,4]-A[2,1:-1,:-2,4])

    x=npr.randn(5)
    Lx=bardensr.preprocessing.preprocessing_tf.grad_sfd(x,[0])
    y=npr.randn(*Lx.shape)
    LTy=bardensr.preprocessing.preprocessing_tf.grad_sfd_transpose(y,[0])
    assert np.allclose(np.sum(Lx*y),np.sum(x*LTy))

    x=npr.randn(4,5,6,7)
    Lx=bardensr.preprocessing.preprocessing_tf.grad_sfd(x,[1,2])
    y=npr.randn(*Lx.shape)
    LTy=bardensr.preprocessing.preprocessing_tf.grad_sfd_transpose(y,[1,2])
    assert np.allclose(np.sum(Lx*y),np.sum(x*LTy))

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

