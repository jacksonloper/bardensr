# adapted from tf code

import collections
import tensorflow as tf

CGState = collections.namedtuple('CGState', ['i', 'x', 'r', 'p', 'gamma'])

def dot(x,y):
    return tf.reduce_sum(x*y)

def cg_steps(cgs,operator,niter,preconditioner=None):
    for i in tf.range(niter):
        cgs=cg_step(cgs,operator,preconditioner)
    return cgs

def cg(operator,rhs,niter,x0=None,cgs0=None,preconditioner=None):
    cgs=initialize_cg(operator,rhs,x0=x0,preconditioner=preconditioner)

    for i in range(niter):
        cgs=cg_step(cgs,operator,preconditioner)
    return cgs

def initialize_cg(operator,rhs,x0=None,preconditioner=None):
    if preconditioner is None:
        p0 = rhs
    else:
        p0 = preconditioner(rhs)

    i=tf.cast(0,dtype=tf.int32)

    if x0 is None:
        x0=tf.zeros_like(rhs)
        r=rhs
    else:
        r=rhs-operator(x0)

    p=(rhs if preconditioner is None else preconditioner(rhs))
    gamma=dot(r,p) # invariant -- gamma = <r | preconditioner | r >

    return CGState(i,x0,r,p,gamma)

def cg_step(cgstate,operator,preconditioner=None,tol=1e-20):
    with tf.name_scope("cg_step"):
        return tf.cond(
            tf.linalg.norm(cgstate.r)<tol,
            lambda: cgstate,
            lambda: _cg_step(cgstate,operator,preconditioner)
        )

def _cg_step(cgstate,operator,preconditioner=None):
    z = operator(cgstate.p)
    alpha = cgstate.gamma / dot(cgstate.p, z)
    x = cgstate.x + alpha * cgstate.p
    r = cgstate.r - alpha * z

    if preconditioner is None:
        q = r
    else:
        q = preconditioner(r)

    gamma = dot(r, q)
    beta = gamma / cgstate.gamma
    p = q + beta * cgstate.p

    return CGState(cgstate.i+1,x,r,p,gamma)
