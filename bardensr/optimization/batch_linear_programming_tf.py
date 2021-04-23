'''
Adapted from scipy implementation of revised simplex method
'''

import tensorflow as tf
import numpy as np
from . import misc


def tal(params,indices,axis=1):
    '''
    out[b,i,...] = params[b,indices[b,i],...]
    '''
    return tf.gather(params,indices,axis,batch_dims=1)


def tal2(params,indices):
    '''
    out[b,i,j] = params[indices[b,i],j]
    '''

    return tf.gather_nd(params,indices)


def tal3(params,indices):
    '''
    out[b,...] = params[b,indices[b],...]
    '''

    # stacked_indices[b,i] = (b,indices[b])
    indices0 = tf.cast(tf.range(tf.shape(params)[0]),dtype=indices.dtype)
    stacked_indices = tf.stack([indices0,indices],axis=1)

    # out[b,i,...] = params[stacked_indices[b,i],...]
    rez=tf.gather_nd(params,stacked_indices)

    return rez

def pal(tensor,indices,updates):
    '''
    tensor[i,indices[i,j]] = updates[i,j]
    '''

    nupdates=tf.reduce_prod(updates.shape)
    ravupt=tf.reshape(updates,(nupdates,))
    ravind=tf.cast(tf.reshape(indices,(nupdates,)),dtype=tf.int64)

    index_batch_numbers = tf.range(indices.shape[0],dtype=tf.int64)
    ones=tf.ones((indices.shape[1],),dtype=tf.int64)
    index_batch_numbers = index_batch_numbers[:,None]*ones[None,:]
    ravibn = tf.reshape(index_batch_numbers,(nupdates,))

    indices2 = tf.stack([ravibn,ravind],axis=-1)

    return tf.tensor_scatter_nd_update(tensor,indices2,ravupt)

def solve_lp_batched(c,A,b,maxiter=100,tol=1e12,use_tqdm_notebook=False):
    '''
    attempts to solve

        min   <c,x>
        s.t.  x>=0
              A@x <=b


    for many values of c,b

    Input:
    - c -- batch x n
    - A -- batch x m x n
    - b -- batch x m

    Output
    - x -- batch x n

    '''

    ######### problem conversion ###########
    '''
    equivalent to this problem:

        min  <c,x>
        s.t. x,y >=0
             A@x + y <=b
    '''
    batch,m,n=A.shape

    A_enlarged = np.concatenate([A,np.eye(m)[None,:,:]*np.ones(batch)[:,None,None]],axis=2)
    c_enlarged = np.concatenate([c,np.zeros((batch,m))],axis=1)
    x0 = np.concatenate([np.zeros((batch,n)),b],axis=1)
    basis=np.ones(batch,dtype=int)[:,None]*(np.r_[n:n+m])[None,:]
    not_basis=np.ones(batch,dtype=int)[:,None]*(np.r_[0:n])[None,:]

    c_enlarged=tf.identity(c_enlarged)
    AT_enlarged=tf.identity(np.transpose(A_enlarged,[0,2,1]))
    x0=tf.identity(x0)
    basis=tf.identity(basis)
    not_basis=tf.identity(not_basis)

    ########## solve the big problem #######
    rez,status=phase_two_batched(c_enlarged,AT_enlarged,x0,
        basis,not_basis,maxiter,1e-12,use_tqdm_notebook)

    ######## return the answer of interest #########
    return rez[:,:n],status


@tf.function
def phase_two_iteration(c,bAT,basis,not_basis,x,tol):
    '''
    let b = AT.T[:,basis] @ x

    returns new and improved basis,not_basis,x,done for the
    objective functions

        min   <c[batch],x[batch]>
        s.t.  x >= 0
              A@x <= b[batch]

    '''

    # compute some things we need
    CURB_AT = tal(bAT,basis) # CURB_AT[b,i,j] = AT[b,basis[b,i],j]
    CURB_c  = tal(c,basis,1) # CURB_c[b,i] = c[b,basis[i]]

    CURNB_c = tal(c,not_basis,1) # CURNB_AT[b,i,j] = AT[b,not_basis[b,i],j]

    #######################
    # look for new guys to bring into basis

    # perform first solve
    v = tf.linalg.solve(CURB_AT,CURB_c[...,None])[...,0]

    # get c_hat on non-basis elts
    # (actually compute the einsum on all elts, cheaper
    # to do this than to do the indexing ahead of time and store
    # a bunch of big matrices!)
    c_hat = CURNB_c - tal(tf.einsum('bij,bj->bi',bAT,v),not_basis)

    # check if we're done
    done=tf.reduce_all(c_hat >= -tol,axis=-1)

    # if one of the coordinates outside the basis
    # is sad, we must bring it into the basis.
    # we will choose the one with lowest c_hat
    j_location_in_not_basis=tf.argmin(c_hat,axis=-1) # batch
    j = tal(not_basis,j_location_in_not_basis[:,None],1)[...,0]

    #######################
    # we will kick somebody out of the basis.  but who?

    # second solve.
    Aj = tal3(bAT,j) # Aj[b,i] = AT[b,j[b],i]
    u = tf.linalg.solve(CURB_AT,Aj[...,None],adjoint=True)[...,0]

    # among guys where u>tol, pick the guy where xb/u is SMALLEST
    CURB_x = tal(x,basis,1)
    bad=u<tol

    th=tf.where(bad,CURB_x.dtype.max,CURB_x/u)
    l = tf.argmin(th,axis=-1)  # NB -- this selects smallest suitable index if there's a tie
    to_remove_location_in_basis=l
    to_remove = tal(basis,to_remove_location_in_basis[:,None],1)[:,0]


    ####################
    # hop to new vertex, while preserving A@x <= b and x>=0
    # only... if statii[b]=0, do nothing!

    # recall xb/u for the guy we're kicking out
    th_star = tal(th,l[:,None],1)

    # make the hop
    new_x=pal(
        x,
        basis,
        CURB_x - th_star*u,
    )
    new_x=pal(
        new_x,
        j[:,None],
        th_star,
    )

    x=tf.where(done[:,None],x,new_x)

    ####################
    # pivot the basis
    # only... if statii[b]=0, do nothing!

    zero=tf.cast(0,dtype=np.int64)
    to_remove_location_in_basis=tf.where(done,zero,to_remove_location_in_basis)
    j=tf.where(done,basis[:,0],j)
    j_location_in_not_basis=tf.where(done,zero,j_location_in_not_basis)
    to_remove=tf.where(done,not_basis[:,0],to_remove)

    # change the basis
    basis = pal(basis,to_remove_location_in_basis[:,None],j[:,None])
    not_basis = pal(not_basis,j_location_in_not_basis[:,None],to_remove[:,None])

    return basis,not_basis,x,done

def phase_two_batched(c, AT, x, basis, not_basis, maxiter, tol,use_tqdm_notebook=False):
    '''
    Input
    - c, batch x n
    - AT, batch x m x n
    - x, batch x n
    - basis, batch x m, integers in {0,1,...,n-1}
    - not_basis, batch x (n-m), integers in {0,1,...,n-1}
    - maxiter, scalar
    - tol, scalar

    let b = A[:,basis] @ x     (where A=AT.T)

    solves

        min   <c[batch],x[batch]>
        s.t.  x >= 0
              A[batch]@x <= b[batch]

    for each batch.

    (note -- this algorithm only works if m<n)

    '''

    for iteration in misc.maybe_trange(maxiter,use_tqdm_notebook):
        basis,not_basis,x,done=phase_two_iteration(c,AT,basis,not_basis,x,tol)


    return x,done
