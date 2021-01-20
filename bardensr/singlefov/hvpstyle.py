import tensorflow as tf
import dataclass

def reconstruction(X,F,codebook,a,b,varphi,**kwargs):
    G = codebook*alpha[:,:,None]  # rcj
    G = tf.einsum('rcj,cC->rCj',G,varphi)

    fact_recon=tf.einsum('xyzj,rcj->xyzrc',F,G)

    recon=fact_recon + a[...,None,None] + b[None,None,None,:,:] # M0 x M1 x M2 x R x C

    return recon

def calc_Fsc(F,codebook,alpha,a,b,varphi,**kwargs):
    G = codebook*alpha[:,:,None]
    G = tf.einsum('rcj,cC->rCj',G,varphi)

    Gsum=tf.reduce_sum(G,axis=0)
    Gsum=tf.reduce_sum(Gsum,axis=0)
    return F*Gsum[None,None,None,:]

def lossinfo(X,F,codebook,alpha,a,b,varphi):
    '''
    X -- M0 x M1 x M2 x R x C
    F -- M0 x M1 x M2 x J
    codebook -- R x C x J
    alpha -- R x C
    a -- M0 x M1 x M2
    b -- R x C
    varphi -- C x C
    '''

    G = codebook*alpha[:,:,None]
    G = tf.einsum('rcj,cC->rCj',G,varphi)

    fact_recon=tf.einsum('xyzj,rcj->xyzrc',F,G)
    l1_loss = tf.reduce_sum(fact_recon)

    recon=fact_recon + a[...,None,None] + b[None,None,None,:,:]

    recon_loss = tf.reduce_sum((recon-X)**2)

    return dict(recon_loss=recon_loss,l1_loss=l1_loss)

def calc_loss(X,F,codebook,alpha,a,b,lam,varphi):
    d=lossinfo(X,F,codebook,alpha,a,b,varphi)
    return d['recon_loss'] + lam*d['l1_loss']


'''

'''


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


@dataclasses.dataclass
class DensityResult:
    density:np.ndarray
    model:Model

def build_density(Xsh,codebook,lam=1.0,use_tqdm_notebook=False,niter=60):
    # Xsh -- R,C,M0,M1,M2
    R,C,M0,M1,M2=Xsh.shape
    J=codebook.shape[-1]

    Xsh=tf.convert_to_tensor(np.transpose(Xsh,[2,3,4,0,1]))

    alpha=tf.zeros((M0,M1,M2,J),dtype=tf.float64)
    a=tf.zeros((M0,M1,M2),dtype=tf.float64)
    b=tf.zeros((R,C),dtype=tf.float64)

    varphi=tf.eye(C,dtype=tf.float64)

    state=dict(
        lam=tf.convert_to_tensor(lam,dtype=tf.float64),
        X=X,
        F=F,
        a=a,
        b=b,
        alpha=alpha,
        varphi=varphi,
        codebook=tf.convert_to_tensor(codebook,dtype=tf.float64),
    )


    if use_tqdm_notebook:
        import tqdm.notebook
        t=tqdm.notebook.trange(niter)
    else:
        t=range(niter)
    for i in t:

        state['F']=improve_F(**state)
        state['a']=improve_a(**state)
        state['b']=improve_b(**state)
        state['alpha']=improve_alpha(**state)

    rez=calc_Fsc(**state).numpy()
    rez=rez/rez.max()

    return DensityResult(density=rez,model=state)

'''

'''


'''

'''

improve_F = cnuwb_speedify(calc_loss,'F')
improve_a = cnuwb_speedify(calc_loss,'a')
improve_alpha = cnuwb_speedify(calc_loss,'alpha',1e-5)
improve_b = cnuwb_speedify(calc_loss,'b')
improve_varphi = cnuwb_speedify(calc_loss,'varphi',1e-5)
