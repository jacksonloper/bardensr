import tensorflow as tf
import contextlib
import collections
from . import cg_tf

@contextlib.contextmanager
def HessianVectorProduct(towatch,search_direction):
    '''
    Input
    - towatch, a tensor
    - searchdir, a tensor of same shape as towatch

    Output is an HVP context manager, which can be used to estimate
    the gradient and the hessian-vector-product in a given direction.
    Example usage:

    with HVP(towatch,searchdir) as h:
        loss = tf.reduce_sum(towatch**2)
        h.process(loss)
    print("(hessian of loss w.r.t. towatch)@searchdir is given by",
            h.hess_vector_product_in_search_direction)
    dst = h.estimated_travelling_distance()
    print("to minimize loss, a reasonable plan is to travel roughly distance",
            dst,"in the direction of the search direction)
    towatch.assign(towatch + searchdir*dst)
    '''
    with tf.autodiff.ForwardAccumulator(towatch,search_direction) as acc:
        with tf.GradientTape() as tape:
            tape.watch(towatch)
            h=_HVP(towatch,search_direction)
            yield h
            if h.loss is None:
                raise Exception("No value to process -- "
                                "call h.process(loss) inside context manager")
        h.grad = tape.gradient(h.loss, towatch)
    h.hess_vector_product_in_search_direction = acc.jvp(h.grad)
    if h.hess_vector_product_in_search_direction is None:
        h.hess_vector_product_in_search_direction=tf.cast(0,towatch.dtype)
    h._mark_complete()

class _HVP:
    def __init__(self,towatch,search_direction):
        self.towatch=towatch
        self.search_direction=search_direction
        self.loss = None
        self._complete=False
        self._gs=None
        self._otd=None

    def _mark_complete(self):
        if self._complete:
            raise Exception("something went weirdly wrong")
        self._complete=True

    @property
    def grad_dot_search_direction(self):
        if not self._complete:
            raise Exception("Cannot compute that yet.  Leave context manager first.")

        if self._gs is None:
            self._gs = tf.reduce_sum(self.search_direction*self.grad)

        return self._gs

    @property
    def estimated_optimal_travelling_distance(self):
        if not self._complete:
            raise Exception("Cannot compute that yet.  Leave context manager first.")

        if self._otd is None:
            hs = tf.reduce_sum(self.hess_vector_product_in_search_direction*self.search_direction)
            self._otd=self.grad_dot_search_direction/hs

        return self._otd

    def process(self,loss):
        if self.loss is None:
            self.loss = loss
        else:
            raise Exception("HVP can only process one value")


def contain_bad_searchdir(val,sd,lo=None,hi=None):
    if (lo is None) and (hi is None):
        return sd

    bad=False
    if lo is not None:
        bad = bad | ((sd<0)&(val<=lo)) # searchdir says go down, but val says no!
    if hi is not None:
        bad = bad | ((sd>0)&(val>=hi)) # searchdir says go up, but val says no!

    # zero out the badness)
    zero=tf.cast(0,sd.dtype)
    search_direction = tf.where(bad,zero,sd)

    return search_direction


ArmijoLinesearchPrep=collections.namedtuple('ArmijoLinesearchPrep',
    [
        'initial_guess',
        'search_direction',
        'travelling_distance',
        'grad',
        'loss',
        'sd_dot_grad',
        'lo',
        'hi',
        'lo_tf',
        'hi_tf',
    ])

def inexact_newton_solve(lossfunc,initial_guess,maxiter=10):

    with tf.GradientTape() as t:
        t.watch(initial_guess)
        loss=lossfunc(initial_guess)
    grad=t.gradient(loss,initial_guess)
    naive_search_direction=-grad

    def operator(search_direction):
        with HessianVectorProduct(initial_guess,search_direction) as h:
            loss = lossfunc(initial_guess)
            h.process(loss)
        return h.hess_vector_product_in_search_direction

    cgs=cg_tf.cg(
        operator,
        naive_search_direction,
        maxiter,
        x0=naive_search_direction,
    ) # solve Hx = -grad

    new_search_direction=cgs.x

    dot=-tf.reduce_sum(new_search_direction*grad)
    tf.debugging.assert_non_negative(dot,summarize=1,
                message="inexact newton solve failed to find a valid searchdir")

    return new_search_direction

def inexact_newton_solve_with_bounds(lossfunc,initial_guess,maxiter=10,lo=None,hi=None):

    with tf.GradientTape() as t:
        t.watch(initial_guess)
        loss=lossfunc(initial_guess)
    grad=t.gradient(loss,initial_guess)
    sd=-grad

    # get the bad places
    bad=False
    if lo is not None:
        bad = bad | ((sd<0)&(initial_guess<=lo)) # searchdir says go down, but val says no!
    if hi is not None:
        bad = bad | ((sd>0)&(initial_guess>=hi)) # searchdir says go up, but val says no!

    # get a reasonable search direction
    zero=tf.cast(0,sd.dtype)
    naive_search_direction = tf.where(bad,zero,sd)

    def operator(search_direction):
        # kill the search direction in bad places
        search_direction_killed = tf.where(bad,zero,search_direction)

        # get hvp in killed direction
        with HessianVectorProduct(initial_guess,search_direction_killed) as h:
            loss = lossfunc(initial_guess)
            h.process(loss)
        hvp=h.hess_vector_product_in_search_direction

        # make operator behave as if the killed directions just stayed put
        return tf.where(bad,search_direction,hvp)

    cgs=cg_tf.cg(
        operator,
        naive_search_direction,
        maxiter,
    ) # solve Hx = -grad

    new_search_direction=cgs.x

    dot=-tf.reduce_sum(new_search_direction*grad)
    tf.debugging.assert_non_negative(dot,summarize=1,
                message="inexact newton solve failed to find a valid searchdir")

    new_search_direction=contain_bad_searchdir(initial_guess,new_search_direction,lo,hi)
    dot=-tf.reduce_sum(new_search_direction*grad)
    tf.debugging.assert_non_negative(dot,summarize=1,
                message="inexact newton solve failed to find a valid searchdir "
                        "(after enforcing bounds...")

    return new_search_direction

def prepare_for_armijo_linesearch(lossfunc,initial_guess,
                    search_direction=None,
                    travelling_distance=None,
                    lo=None,
                    hi=None,
                    enforce_bounds=True,
                    enforce_bounds_sd=True,
                ):
    '''
    Input:
    - lossfunc, a callable with one argument
    - initial_guess, a tensor of correct shape/dtype to be supplied to lossfunc
    - [optional] search_direction, a direction to look in (use gradient by default)
    - [optional] travelling_distance, an initial guess for how far to go
                        (quadratic approximation used by default)
    - [optional] lo, a lower bound for variable
    - [optional] hi, an upper bound for variable
    - [optional] enforce_bounds -- whether or not to enforce hi and lo on initial guess
    - [optional] enforce_bounds_sd -- whether or not to enforce that search direction
                                        won't immediately violate bounds

    Output is ArmijoLinesearchPrep, an argument which can be supplied
    to armijo_linesearch to do the actual search
    '''

    lo_tf = initial_guess.dtype.min if (lo is None) else tf.cast(lo,initial_guess.dtype)
    hi_tf = initial_guess.dtype.max if (hi is None) else tf.cast(hi,initial_guess.dtype)

    if enforce_bounds:
        initial_guess = tf.clip_by_value(initial_guess,lo_tf,hi_tf)

    # get gradient
    with tf.GradientTape() as t:
        t.watch(initial_guess)
        loss=lossfunc(initial_guess)
    grad=t.gradient(loss,initial_guess)

    # pick searchdir
    if search_direction is None:
        search_direction=-grad
        search_direction=contain_bad_searchdir(initial_guess,search_direction,lo,hi)
    elif enforce_bounds_sd:
        dot=tf.reduce_sum(search_direction*grad)
        tf.debugging.assert_non_positive(dot,summarize=1,
                message="this is a terrible search direction for minimizing lossfunc "
                        "maybe you are trying to maximize instead of minimize?")

        search_direction=contain_bad_searchdir(initial_guess,search_direction,lo,hi)

        dot=tf.reduce_sum(search_direction*grad)
        tf.debugging.assert_non_positive(dot,summarize=1,
                message="after removing vectors pointing in the wrong direction "
                        "this has become a terrible search direction.  pick a better one")

    # pick travelling distance guess
    if travelling_distance is None:
        with HessianVectorProduct(initial_guess,search_direction) as h:
            loss = lossfunc(initial_guess)
            h.process(loss)
        travelling_distance=h.estimated_optimal_travelling_distance
        sd_dot_grad=h.grad_dot_search_direction
    else:
        sd_dot_grad = tf.reduce_sum(search_direction*grad)

    return ArmijoLinesearchPrep(
        initial_guess,
        search_direction,
        travelling_distance,
        grad,
        loss,
        sd_dot_grad,
        lo,
        hi,
        lo_tf,
        hi_tf,
    )

def armijo_linesearch(lossfunc,prep,armijo_c1=1e-4,decay_rate=.5,maxiter=10):
    '''
    Input:
    - lossfunc, a function with one argument (to be minimized)
    - prep, an ArmijoLinesearchPrep object created by prepare_for_armijo_linesearch
    - [optional] decay_rate -- how quickly to decay in line search
    - [optional] armijo_c1 -- if this armijo is reached, stop
    - [optional] maxiter -- if maxiter of line search is reached, stop

    Output:
    - new_guess, tensor, a new choice for input to lossfunc, one which yields a loss
            which is either the same or lower than the loss for the initial guess
    - new_loss, value of loss at new point
    - armijo_constant, how much armijo we were able to achieve, i.e.
            (f(x+delta) - f(x)) / delta@grad
        if delta@grad is zero, we adopt convention that 0/0=0
    - niter, int, how many iterations we used
    '''

    if prep.sd_dot_grad==0:
        zero=tf.cast(0,dtype=prep.initial_guess.dtype)
        return prep.initial_guess,prep.loss,zero,0

    # do the loop
    force=1.0
    for i in range(maxiter):
        # get new guess
        newguess = prep.initial_guess - force*prep.travelling_distance*prep.search_direction

        # project to bounds
        if (prep.lo is not None) or (prep.hi is not None):
            newguess = tf.clip_by_value(newguess,prep.lo_tf,prep.hi_tf)

        # see how good it is
        newloss = lossfunc(newguess)

        # get armijo constant
        armijo = (newloss - prep.loss) / (force*prep.sd_dot_grad)

        if armijo>armijo_c1:
            break
        else:
            force=force*decay_rate

    if newloss<prep.loss:
        return newguess,newloss,armijo,i
    else:
        zero=tf.cast(0,dtype=prep.initial_guess.dtype)
        return prep.initial_guess,prep.loss,zero,i
