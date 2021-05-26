
import numpy as np
import scipy as sp
import scipy.ndimage
import tensorflow as tf


import logging
logger = logging.getLogger(__name__)


def optional(x,y,func):
    if x is None:
        return func(y,dtype=tf.float64)
    else:
        rez=tf.convert_to_tensor(x,dtype=np.float64)
        assert rez.shape==tuple(y)
        return rez

def optional_const(x,y):
    if x is None:
        return y
    else:
        rez=tf.convert_to_tensor(x,dtype=np.float64)
        assert rez.shape==y.shape
        return rez


def optional_eye(x,y):
    if x is None:
        return tf.eye(y,dtype=tf.float64)
    else:
        rez=tf.convert_to_tensor(x,dtype=np.float64)
        assert rez.shape==(y,y)
        return rez

def phasing(B,q):
    R,C,J=B.shape
    Z=[B[0]]
    for r in range(1, R):
        Z.append(q[:, None]*Z[r-1] + B[r])
    return tf.stack(Z,axis=0)

def nonnegative_update(apply_Gamma,phi,y,lo=0,backtracking=.9,maxiter=10):
    '''
    Consider the problem of optimizing

        .5* y^T Gamma y - phi^T y

    subject to the constraint that (y>=lo).all()

    given an initial guess for y,
    this function finds a new value for y,
    which is gauranteed to be Not Worse!
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
