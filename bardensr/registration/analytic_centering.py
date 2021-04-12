import numpy as np
import scipy as sp
import scipy.optimize

def analytic_center(A,b,eps,x0,method='nelder-mead'):
    '''
    Input:
    - A   -- N x M
    - b   -- N
    - eps -- N
    - x0  -- M

    Output is a scipy OptimizeResult for

       max np.sum(eps*log(b-A@x))

    optimized using x0 as a starting point

    '''

    def func(x):
        if ((b-A@x)<=0).any():
            return np.inf
        return -np.sum(eps*np.log(b-A@x))

    def jac(x):
        return A.T@(eps/(b-A@x))

    if method=='nelder-mead':
        return sp.optimize.minimize(func,x0,method=method)
    else:
        return sp.optimize.minimize(func,x0,jac=jac,method=method)


def construct_reasonable_rectangle_problem(affines,epsilon=.1,maxsize=300):
    affines=np.require(affines).astype(float)
    n=affines.shape[0]
    assert affines.shape==(n,n)
    maxsize=np.require(maxsize).astype(float)
    if maxsize.shape==():
        maxsize=np.ones(n)*maxsize

    hopefully_feasible=np.ones(n)*5

    affines=np.sign(np.sum(affines,axis=1,keepdims=True)) * affines

    A=np.concatenate([
        -np.eye(n),  # -x <=0
        affines,     # A@x < 1
        np.eye(n),   # x < maxsize
    ],axis=0)
    b=np.concatenate([
        np.zeros(n),
        np.ones(n),
        maxsize
    ],axis=0)

    if not (A@hopefully_feasible<b).all():
        raise ValueError("affine transformation is too big, try another method")


    eps=np.r_[np.ones(n),np.ones(2*n)*epsilon]

    return A,b,eps,hopefully_feasible

def calc_reasonable_rectangles(affines,epsilon=.01,maxsize=300):
    '''
    Input
    - affines -- N
    - epsilon -- scalar
    - maxsize -- scalar or N

    Output
    - rectangle -- N, integer

    Finds a reasonably large value of rectangle so that

      (np.abs(affines@rectangle)<1).all()
      (rectangle<maxsize).all()

    We do this in two steps.  First, we get all the
    affines pointing in the same direction.

        affines=np.sign(np.sum(affines,axis=1,keepdims=True)) * affines

    and then solving the optimization problem

      max_rectangle
          np.sum(np.log(rectangle))
        + epsilon*np.sum(np.log(maxsize-rectangle))
        + epsilon*np.sum(np.log(b-affines@rectangle))
    '''

    A,b,eps,hopefully_feasible = construct_reasonable_rectangle_problem(
        affines,epsilon,maxsize
    )

    rez=analytic_center(A,b,eps,hopefully_feasible,method='nelder-mead')
    return np.floor(rez['x']).astype(int)
