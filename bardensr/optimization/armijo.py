import tensorflow as tf
import contextlib
import collections


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
        'initial_loss',
        'sd_dot_grad',
        'lo',
        'hi',
        'lo_tf',
        'hi_tf',
    ])


def prepare_for_armijo_linesearch_on_quadratic_objective(psd_op,intercept,initial_guess,
                    travelling_distance=None,
                    lo=None,
                    hi=None,
                    enforce_bounds=True,
                ):

    Tx0=psd_op(initial_guess)
    x0b=tf.reduce_sum(initial_guess*intercept)
    initial_loss=.5*tf.reduce_sum(Tx0*initial_guess) - x0b

    lo_tf = initial_guess.dtype.min if (lo is None) else tf.cast(lo,initial_guess.dtype)
    hi_tf = initial_guess.dtype.max if (hi is None) else tf.cast(hi,initial_guess.dtype)

    if enforce_bounds:
        initial_guess = tf.clip_by_value(initial_guess,lo_tf,hi_tf)

    # get gradient of objective .5<x | psd_op | x> - <x, intercept>
    # which is just psd_op@x0 - intercept
    grad = Tx0 - intercept
    sd_dot_grad=-tf.reduce_sum(grad**2)

    search_direction=-grad
    search_direction=contain_bad_searchdir(initial_guess,search_direction,lo,hi)

    # pick travelling distance guess
    if travelling_distance is None:
        # |grad|^2 / <grad | psd_op | grad>
        travelling_distance = -sd_dot_grad / tf.reduce_sum(grad*psd_op(grad))

    return ArmijoLinesearchPrep(
        initial_guess,
        search_direction,
        travelling_distance,
        grad,
        initial_loss,
        sd_dot_grad,
        lo,
        hi,
        lo_tf,
        hi_tf,
    )


ArmijoLinesearchResult=collections.namedtuple('ArmijoLinesearchResult',
    [
        'newguess',
        'newloss',
        'armijoconstant',
        'niter'
    ])

def armijo_linesearch(lossfunc,prep,armijo_c1=1e-4,decay_rate=.5,maxiter=10):
    '''
    Input:
    - lossfunc, thing we're trying to optimize
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
        return ArmijoLinesearchResult(prep.initial_guess,prep.initial_loss,zero,0)

    # do the loop
    force=1.0
    for i in range(maxiter):
        # get new guess
        newguess = prep.initial_guess + force*prep.travelling_distance*prep.search_direction

        # project to bounds
        if (prep.lo is not None) or (prep.hi is not None):
            newguess = tf.clip_by_value(newguess,prep.lo_tf,prep.hi_tf)

        # see how good it is
        newloss = lossfunc(newguess)

        # get armijo constant
        armijo = (newloss - prep.initial_loss) / (force*prep.travelling_distance*prep.sd_dot_grad)

        if armijo>armijo_c1:
            break
        else:
            force=force*decay_rate

    if newloss<prep.initial_loss:
        return ArmijoLinesearchResult(newguess,newloss,armijo,i)
    else:
        zero=tf.cast(0,dtype=prep.initial_guess.dtype)
        return ArmijoLinesearchResult(prep.initial_guess,prep.initial_loss,zero,i)