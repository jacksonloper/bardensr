import tensorflow as tf

def clever_nonneg_update_with_backtracking_precomputation(lossfunc,F_guess,lo=0.0):
    # get a smart direction
    with tf.GradientTape() as tape:
        tape.watch(F_guess)
        loss0 = lossfunc(F_guess)
        search_direction = tape.gradient(loss0, F_guess)

    # find gradients which are pointing in stupid direction (i.e. search_direction[i]>0 and F_guess[i]=0)
    bad = (search_direction>lo)&(F_guess<=lo)
    zeros=tf.zeros(tf.ones(len(F_guess.shape),dtype=tf.int32),dtype=F_guess.dtype)

    # zero those out
    search_direction = tf.where(bad,zeros,search_direction)

    # get hvp
    with tf.autodiff.ForwardAccumulator(F_guess,search_direction) as acc:
        with tf.GradientTape() as tape:
            tape.watch(F_guess)
            loss0 = lossfunc(F_guess)
        grad0 = tape.gradient(loss0, F_guess)
    grad0_vector= tf.reduce_sum(search_direction*grad0)
    hess0_vector= acc.jvp(grad0)
    hess0_vectorvector = tf.reduce_sum(hess0_vector*search_direction)

    # travelling distance
    travelling_distance=grad0_vector/hess0_vectorvector

    return loss0,search_direction,travelling_distance

def clever_nonneg_update_with_backtracking_from_precomputation(
                lossfunc,F_guess,loss0,search_direction,travelling_distance,maxiter=10,decay=.5,force=1.0,lo=0.0):

    zeros=tf.ones(tf.ones(len(F_guess.shape),dtype=tf.int32),dtype=F_guess.dtype)*lo

    improved=False
    for i in range(maxiter):
        newguess = F_guess - force*travelling_distance*search_direction

        # project to positivitytown
        newguess = tf.where(newguess<lo,zeros,newguess)

        newloss = lossfunc(newguess)

        if newloss<loss0: # we improved!  let's call it a day
            return newguess
        else: # we didn't improve.  let's try again with less force
            force=force*decay

    # even after maxiter, we couldn't do better. give up.
    return F_guess

def cnuwb_speedify(lossfunc,nm,lo=0.0):
    def uk(nm,Fg,**kwargs):
        kwargs=dict(kwargs)
        kwargs.update({nm:Fg})
        return kwargs

    @tf.function(autograph=False)
    def improve_F_precompute(**kwargs):
        lfl=lambda Fg: lossfunc(**uk(nm,Fg,**kwargs))
        return clever_nonneg_update_with_backtracking_precomputation(lfl,kwargs[nm],lo=lo)

    def improve_F(**kwargs):
        lfl=lambda Fg: lossfunc(**uk(nm,Fg,**kwargs))
        pc=improve_F_precompute(**kwargs)
        return clever_nonneg_update_with_backtracking_from_precomputation(lfl,kwargs[nm],*pc,lo=lo)

    return improve_F
