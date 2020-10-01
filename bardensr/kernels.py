import tensorflow as tf

def heat_kernel(X,niter,axis=0):
    pad_width=[(0,0) for i in range(len(X.shape))]
    pad_width[axis]=(niter,niter)
    pad_width=tf.convert_to_tensor(pad_width)
    X=tf.pad(X,pad_width)
    sl=[slice(0,None) for i in range(len(X.shape))]
    sl1=list(sl); sl1[axis]=slice(1,None)
    sl2=list(sl); sl2[axis]=slice(None,-1)
    for i in range(niter*2):
        X=.5*(X[sl1]+X[sl2])
    return X


# @tf.function(autograph=False)
def heat_kernel_nd(X,niters):
    for i in range(len(niters)):
        if niters[i]>0:
            X=heat_kernel(X,niters[i],axis=i)
    return X
