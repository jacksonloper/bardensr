import tensorflow as tf

def evaluate_histogram_as_density(edges,counts,x):
    '''
    Takes a probability density function, represented as a histogram,
    and evaluates it on the values in x.
    '''

    # normalize counts
    pmf=counts/tf.reduce_sum(counts)

    # flatten
    flat=tf.reshape(x,tf.reduce_prod(tf.shape(x)))

    # decide which bin to put each x in
    bin_idxs=tf.searchsorted(edges,flat)

    # index into pmf
    probs = tf.gather(pmf,bin_idxs)

    # unflatten
    return tf.reshape(probs,x.shape)

def construct_1dslice(shp,st,sz,axis):
    '''
    Input:
    shp -- result of tf.shape
    st -- scalar tf.int32
    sz -- scalar tf.int32
    axis -- scalar tf.int32

    Output:
    stv
    szv

    Such that
    stv[i] = st if i==axis else 0
    szv[i] = -1 if i==axis else shp[i]
    '''

    nd=tf.shape(shp)

    # make stv
    stv=tf.scatter_nd([[axis]],[st],nd)

    # make szv
    szv=tf.cast(tf.fill(nd,-1),dtype=tf.int32)
    szv=tf.tensor_scatter_nd_update(szv,[[axis]],[sz])

    return stv,szv