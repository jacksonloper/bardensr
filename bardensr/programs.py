
import numpy as np
import scipy as sp
import scipy.ndimage
import tensorflow as tf


import logging
logger = logging.getLogger(__name__)

def solve_square_nnls(Xtilde_against_Xtilde,X_against_Xtilde,Xtilde_rowsums,lam,eps=1e-10):
    r'''
    Consider the loss

    $$L(\varphi) = \frac{1}{2} \sum_{mn} \left(\mathbf{X}_{mn} - \sum_{n'} \mathbf{\tilde{X}}_{m,n'} \varphi_{n,n'} \right)^2 + \lambda \sum_{mn}\sum_{n'}  \mathbf{\tilde{X}}_{m,n'} \varphi_{n,n'} + \frac{1}{2}\epsilon \sum_{n,n'} \varphi_{n,n'}^2$$

    This function uses sp.optimize.nnls to find the best possible value for $\varphi$.  This function is slow if $n$ is large.

    Input:
    - Xtilde_against_Xtilde -- a (N x N) tf2 matrix giving Xtilde.T @ Xtilde
    - X_against_Xtilde -- a (N x N) tf2 matrix giving X.T @ Xtilde
    - Xtilde_rowsums -- a (N) tf2 vector giving sum(Xtilde,axis=0)
    - lam -- scalar
    - [optional] -- eps=1e-10 (for numerical stability)

    Output: best possible varphi
    '''
    raise Exception("NYI")


    n=Xt.shape[1]

    Gamma = Xtilde_against_Xtilde.numpy()
    phi = X_against_Xtilde.numpy() - lam*Xtilde_rowsums.numpy()[:,None]
    A=sp.linalg.cholesky(Gamma+np.eye(Gamma.shape[0])*eps,lower=False)

    varphi=np.ones(n)
    for n in range(Xt.shape[1]):
        b = np.linalg.solve(A.t@phi[n])
        varphi[n]= sp.optimize.nnls(A,b)[0]
    return tf.convert_to_tensor(varphi,dtype=tf.float64)

@tf.function(autograph=False)
def KKGTG(x,K,Gt,G):
    '''
    returns K@K@x@G.T@G
    '''
    return K(K(x@Gt)) @ G

def calc_matrix_nnls_loss(X,G,F,K,lam):
    Xt = K(F)@tf.transpose(G)

    recon = .5*tf.reduce_sum((X-Xt)**2)
    l1 = lam*tf.reduce_sum(Xt)

    return recon+l1

def iterate_matrix_nnls(X,G,F,K,lam,**kwargs):
    r'''
    Consider the loss

    $$L(\mathbf{F}) = \frac{1}{2} \sum_{mn} \left(\mathbf{X}_{mn} -\sum_{i,k} \mathbf{G}_{nk} \mathbf{F}_{i k} \mathbf{K}_{m,i}\right)^2 + \lambda \sum_{mn}\sum_{i,k} \mathbf{G}_{nk} \mathbf{F}_{i k} \mathbf{K}_{m,i}$$
    Given an initial nonnegative guess for $\mathbf{F}$, this function makes a new guess for
    F that is nonnegative and doesn't make the loss worse (and might make it better).

        Input:
        - X -- a (M x N) tf2 tensor
        - G -- a (N x J) tf2 tensor
        - F -- a (M x J) tf2 tensor
        - K -- a callable that takes a tf2 matrix of shape (M \times ?) and returns a tf2 matrix of shape (M \times ?)
        - lam (scalar)
        - [optional] backtracking=.9
        - [optional] maxiter=10

        The callable K should apply a symmetric linear transformation to each row of its input.

        See nonnegative_quadratic_iteration for details on the meaning of backtracking and maxiter

        Output: a new guess for F

    '''
    framel1 = tf.reduce_sum(G,axis=0)
    framel2 = tf.reduce_sum(G**2,axis=0)
    xmabl=X - lam
    linear_term = K(xmabl) @ G
    Gt=tf.transpose(G)
    def apply_Gamma(x):
        return KKGTG(x,K,Gt,G)

    return iterate_nonnegative_quadratic(apply_Gamma,linear_term,F,**kwargs)

def iterate_nonnegative_quadratic(apply_Gamma,phi,y,lo=0,backtracking=.9,maxiter=10):
    r'''
    Consider the problem of optimizing

    $$L(y)=\frac{1}{2} \sum_{ij} y_i y_j \Gamma_{ij} - \sum_i y_i \phi_i$$

    subject to the constraint that $y_i>=\mathtt{lo}$ for every $i$.

    The function `nonnegative_quadratic_iteration` helps with this problem.  Given an initial guess $y$,
    this function finds a new value $y'$ such that $L(y')\leq L(y)$.

    The algorithm works by making a cleverly chosen guess at a search direction and
    a cleverly chosen guess about how far to go in that direction.   If going that far
    makes the objective worse, we try going a little less far (this is called
    backtracking).  If it still doesn't work we back track a little more.  If it
    still doesn't work we backtrack yet more.  We do this a couple more times and then
    give up.


    ```
    Input:
    - apply_Gamma -- a callable that takes a tf2 vector y and returns a new vector with the same shape
    - phi -- a tf2 vector
    - y -- a tf2 vector which is the same shape as phi
    - [optional] lo=0
    - [optional] backtracking=.9 --  how much to backtrack.  For example, backtracking=.9 means if
    at first we don't succeed we will try going only 90% of the distance we went at first.
    - [optional] maxiter=10 -- we repeat the backtracking process "maxiter" iterations until
    we make some improvement or until maxiter is reached.

    The callable apply_Gamma should apply a symmetric positive definite linear transformation to its input.

    Ouput: a new guess for y.  Designed to reduce the loss function
      loss(y) = .5*tf.reduce_sum(y*apply_Gamma(y)) - tf.reduce_sum(y*phi)
    ```
    '''

    gY=apply_Gamma(y)

    # get a search direction
    search_dir = phi - gY

    # but we shouldn't go negative if we have active constraints!
    bad_id = (y <= lo)&(search_dir <= 0)
    zero=tf.zeros(search_dir.shape,dtype=search_dir.dtype)
    search_dir=tf.where(bad_id,zero,search_dir)

    # assuming there are at least some nontrivial directions, do something
    if tf.reduce_sum(search_dir**2)>1e-10:
        # get old loss
        old_loss = .5*tf.reduce_sum(y*gY) - tf.reduce_sum(phi*y)

        # how far should we go in this modified search direction?
        Gamma_search_dir=apply_Gamma(search_dir)
        bunsi = tf.reduce_sum(phi*search_dir) - tf.reduce_sum(y*Gamma_search_dir)
        bunmu = tf.reduce_sum(search_dir * Gamma_search_dir)
        lr= bunsi/bunmu

        proposed_y = tf.clip_by_value(y+lr*search_dir,lo,np.inf)
        new_loss = .5*tf.reduce_sum(proposed_y*apply_Gamma(proposed_y)) - tf.reduce_sum(phi*proposed_y)

        # backtrack as necessary
        for i in range(maxiter):
            if new_loss<old_loss: # we made progress!  done!
                return proposed_y
            else:  # we didn't.  lets try to backtrack a bit
                logger.info("backtracking")
                lr=lr*backtracking
                proposed_y = tf.clip_by_value(y+lr*search_dir,lo,np.inf)
                new_loss = .5*tf.reduce_sum(proposed_y*apply_Gamma(proposed_y)) - tf.reduce_sum(phi*proposed_y)

        # even after backtracking a bunch of times
        # we still seem to be just making things worse.
        # abort!
        return y
    else:
        return y
