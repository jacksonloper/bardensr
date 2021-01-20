import numpy as np
import scipy as sp
import scipy.ndimage
import dataclasses

import scipy.optimize
import scipy.linalg
import numpy.linalg
import re
import numpy.random as npr

from . import helpers
from . import helpers_tf

from .. import kernels

import tensorflow as tf

import logging
logger = logging.getLogger(__name__)

class HeatKernel:
    def __init__(self,spatial_dims,blur_level):
        self.spatial_dims=spatial_dims
        self.nspatial=len(self.spatial_dims)

        if blur_level is None:
            self.blur_level=None
        elif isinstance(blur_level,int):
            self.blur_level=np.ones(self.nspatial,dtype=np.int)*blur_level
        else:
            self.blur_level=np.array(blur_level)
            assert self.blur_level.dtype==np.int64
            assert self.blur_level.shape==(self.nspatial,)

    def __matmul__(self,X):
        if self.blur_level is None:
            return X
        else:
            bl=tuple([int(b) for b in self.blur_level])
            return kernels.heat_kernel_nd(X,bl)

class Model:
    def __init__(self,codebook,spatial_dims,blur_level=None,F=None,a=None,b=None,alpha=None,rho=None,varphi=None,
                        lo=1e-10,lam=0):
        '''
        A Model object holds the parameters for our model

        Input:
        - codebook -- binary codebook (R x C x J)
        - spatial_dims -- i.e. (npix_X,npix_Y) for 2d data or (npix_X,npix_Y,npix_Z) for 3d data
        - [optional] blur_level -- how many iterations of blur
        - [optional] F -- ndarray of shape spatial_dims
        - [optional] a -- ndarray of shape spatial_dims
        - [optional] b -- ndarray of shape R x C
        - [optional] alpha -- ndarray of shape R x C
        - [optional] rho -- ndarray of shape C
        - [optional] varphi -- ndarray of shape C x C
        - [optional] lo -- scalar, smallest possible value of alpha
        - [optional] lam -- magnitude of L1 penalty on gene reconstruction

        If the optional parameters are not given, they will be initialized
        automatically.  One caveat to this is F and M -- one or the other
        must be provided:
        - If F is given then M is not required.
        - If M is given then F is not required.
        - If both are given their shapes should agree.
        - If neither are given an exception will be thrown.
        '''

        self.codebook=tf.convert_to_tensor(codebook)
        self.spatial_dims=tuple(spatial_dims)
        assert len(self.spatial_dims) in [1,2,3]
        self.nspatial=len(self.spatial_dims)

        self.R,self.C,self.J=self.codebook.shape

        self.blur_level=blur_level
        self.K=HeatKernel(spatial_dims,blur_level)

        self.lo=lo
        self.lam=lam

        if len(self.codebook.shape)!=3:
            B_shape_error_message=fr'''
                B is expected to be a 3-dimensional boolean numpy array.
                B[r,c,j] is supposed to indicate whether gene "j" should appear
                bright in round "r" and channel "c".  Instead, we got an object
                with shape {B.shape} and type {B.dtype}
            '''
            raise ValueError(helpers.kill_whitespace(B_shape_error_message))
        self.codebook=tf.cast(self.codebook,dtype=tf.float64)

        # handle all the other initializations
        self.F=helpers_tf.optional(F,self.spatial_dims+(self.J,),tf.zeros)
        self.a=helpers_tf.optional(a,(self.spatial_dims),tf.zeros)
        self.b=helpers_tf.optional(b,(self.R,self.C),tf.zeros)
        self.alpha=helpers_tf.optional(alpha,(self.R,self.C),tf.ones)
        self.varphi=helpers_tf.optional_eye(varphi,self.C)
        self.rho=helpers_tf.optional(rho,(self.C,),tf.zeros)

        # calc some things we'll need later
        self.M=np.prod(self.spatial_dims)
        self.N=self.R*self.C
        self.F_blurred=self.K@self.F
        self.nobs = self.M*self.N

    # code for saving parameters
    _props = ['codebook','spatial_dims','blur_level','F','a','b','alpha','rho','varphi','lo','lam']
    def snapshot(self):
        def npify(x):
            if tf.is_tensor(x):
                return x.numpy().copy()
            else:
                return x

        return {x:npify(getattr(self,x)) for x in self._props}
    def copy(self):
        return Model(**self.snapshot())

    # intensity scaled to show total contribution of a gene to the original images
    def F_scaled(self,blurred=False):
        framel1=tf.reduce_sum(tf.reduce_sum(self.frame_loadings(),axis=0),axis=0)
        if blurred:
            return (framel1[None,:] * self.F_blurred).numpy()
        else:
            return (framel1[None,:] * self.F).numpy()

    # reconstructions
    def Z(self):
        return helpers_tf.phasing(self.codebook,self.rho)
    def frame_loadings(self):
        return tf.einsum('rc,ck, rkj -> rcj',self.alpha,self.varphi,self.Z())
    def gene_reconstruction(self,rho=None,alpha=None,varphi=None):
        frame_loadings = self.frame_loadings()
        return tf.einsum('...j,rcj->...rc',self.F_blurred,frame_loadings)
    def a_broadcast(self):
        sl = (len(self.spatial_dims)*(slice(0,None),)) + ((None,)*2)
        return self.a[sl]
    def b_broadcast(self):
        sl = (len(self.spatial_dims)*(None,)) + ((slice(0,None),)*2)
        return self.b[sl]
    def ab_reconstruction(self):
        return self.a_broadcast() + self.b_broadcast()
    def reconstruction(self):
        return self.ab_reconstruction()+self.gene_reconstruction()
    def FbmixedZ(self):
        '''
        FbmixedZ[m,r,c] = sum_jc' F_blurred[m,j] * varphi[c,c'] * Z[r,c',j]
        '''
        mixedZ =tf.einsum('ck, rkj -> rcj',self.varphi,self.Z())
        FbmixedZ = tf.einsum('rcj,...j -> ...rc',mixedZ,self.F_blurred)
        return FbmixedZ

    # loss
    def loss(self,X):
        ab_recon = self.ab_reconstruction() # a1 + 1b
        gene_recon = self.gene_reconstruction() # KFG

        reconstruction_loss = .5*tf.reduce_sum((X-ab_recon - gene_recon)**2).numpy()
        l1_loss = tf.reduce_sum(gene_recon).numpy()  # L1_loss = |KFG^T|_1

        lossinfo= dict(
            reconstruction = reconstruction_loss,
            l1 = l1_loss,
            lam=self.lam,
        )
        lossinfo['l1_times_lam']=self.lam*lossinfo['l1']
        lossinfo['total_loss']=lossinfo['reconstruction'] + lossinfo['l1_times_lam']
        lossinfo['loss'] = lossinfo['total_loss']/self.nobs

        return lossinfo

    # the updates!
    def update_a(self,X):
        resid = X - (self.gene_reconstruction() + self.b_broadcast()) # spatial dims x R x C
        resid = tf.reduce_mean(tf.reduce_mean(resid,axis=-1),axis=-1) # spatial_dims
        self.a = tf.clip_by_value(resid,0,np.inf) # spatial_dims

    def update_b(self,X):
        resid = X - (self.gene_reconstruction() +self.a_broadcast()) # spatial_dims x R x C
        for i in range(len(self.spatial_dims)):
            resid=tf.reduce_mean(resid,axis=0)
        self.b=tf.clip_by_value(resid,0,np.inf)  # R x C

    def apply_Gamma(self,x,Gt,G):
        return (self.K @ (self.K @ (x@Gt))) @ G

    def update_F(self,X):
        G = tf.reshape(self.frame_loadings(),(self.N,self.J))
        framel1 = tf.reduce_sum(G,axis=0)
        framel2 = tf.reduce_sum(G**2,axis=0)
        xmabl=tf.reshape(X - self.ab_reconstruction() - self.lam,self.spatial_dims+(self.N,))

        '''
        loss = .5* ||X - ab - KFG^T ||^2 + lam*||KFG^T||_1
             = .5* ||KFG^T||^2 - tr((KFG^T) (X - ab - lam)^T)
             = .5* tr(KFG^T G F^T K) - tr(F G (X - ab - lam)^T K)
        '''

        linear_term = (self.K@ xmabl) @ G

        def apply_Gamma(x):
            return self.apply_Gamma(x,tf.transpose(G),G)

        self.F = helpers_tf.nonnegative_update(apply_Gamma,linear_term,self.F)
        self.F_blurred = self.K@self.F

    def update_alpha(self,X):
        # get the update
        Xmabl = tf.reshape(X - self.ab_reconstruction() - self.lam,(self.M,self.R,self.C))
        FbmixedZ=tf.reshape(self.FbmixedZ(),(self.M,self.R,self.C))
        numerator = tf.einsum('mrc,mrc->rc',FbmixedZ,Xmabl)
        denom = tf.reduce_sum(FbmixedZ**2,axis=0)

        # handle possibly zero denominators
        good = denom>self.lo
        alpha=tf.where(good,numerator/denom,self.alpha)

        # clip
        self.alpha=tf.clip_by_value(alpha,self.lo,np.inf)

    def update_varphi(self,X):
        Z=self.Z() # R x C x J
        xmabl = X - self.ab_reconstruction() - self.lam # spatial x R x C
        F=self.F_blurred # spatial x J

        xmabl=tf.reshape(xmabl,(self.M,self.R,self.C))
        F=tf.reshape(F,(self.M,self.J))

        FZ = tf.einsum('mj,rcj->mrc',F,Z)
        FZ_gamma = tf.einsum('mrc,mrk->rck',FZ,FZ)

        varphi=self.varphi.numpy()
        for c1 in range(self.C):
            Gamma_c = tf.einsum('r,rck->ck',self.alpha[:,c1]**2,FZ_gamma).numpy()
            phi_c = tf.einsum('r,mr,mrc->c',self.alpha[:,c1],xmabl[:,:,c1],FZ).numpy()
            A,b=helpers.quadratic_form_to_nnls_form(Gamma_c,phi_c)
            varphi[c1]= sp.optimize.nnls(A,b)[0]
        self.varphi=tf.convert_to_tensor(varphi,dtype=tf.float64)

    def update_rho(self,X):
        pass



@tf.function(autograph=False)
def gaussian_filter_3d(X,sigmas):
    '''
    X -- ... x M0 x M1 x M2
    sigma -- tuple of length 3
    '''

    nd=len(X.shape)
    X=gaussian_filter_1d(X,sigmas[0],nd-3)
    X=gaussian_filter_1d(X,sigmas[1],nd-2)
    X=gaussian_filter_1d(X,sigmas[2],nd-1)

    return X

def gaussian_filter_1d(X,sigma,axis):
    '''
    X -- tensor
    sigma -- scalar
    axis

    filters X over axis
    '''
    xs=tf.cast(tf.range(-sigma*3+1,sigma*3+2),dtype=X.dtype)
    filt=tf.math.exp(-.5*xs**2/(sigma*sigma))
    filt=filt/tf.reduce_sum(filt)
    filt=filt[:,None,None] # width x 1 x 1

    # now we got to transpose X annoyingly

    axes=list(range(len(X.shape)))
    axes[-1],axes[axis]=axes[axis],axes[-1]

    X_transposed=tf.transpose(X,axes) # everythingelse x axis x 1

    newshp=(np.prod(X_transposed.shape[:-1]),X_transposed.shape[-1],1)
    X_transposed_reshaped=tf.reshape(X_transposed,newshp)

    X_convolved=tf.nn.conv1d(X_transposed_reshaped,filt,1,'SAME')
    X_convolved_reshaped=tf.reshape(X_convolved,X_transposed.shape)

    X_convolved_reshaped_transposed=tf.transpose(X_convolved_reshaped,axes)

    return X_convolved_reshaped_transposed

def doublenorm(X,lowg=1,sigma=5):
    R,C,M0,M1,M2=X.shape

    X=X/X.max()
    X=np.reshape(X,(R*C,M0,M1,M2))
    X_bl=gaussian_filter_3d(X,(sigma,sigma,sigma)).numpy().reshape(X.shape)

    return (X/(lowg+X_bl)).reshape((R,C,M0,M1,M2))

@dataclasses.dataclass
class DensityResult:
    density:np.ndarray
    model:Model
    X:np.ndarray

def build_density(Xsh,codebook,lam=.01,use_tqdm_notebook=False,niter=120,blur_level=1):
    # Xsh -- R,C,M0,M1,M2
    Xsh=Xsh/Xsh.max()

    Xsh=tf.convert_to_tensor(np.transpose(Xsh,[2,3,4,0,1]))

    m=Model(codebook,Xsh.shape[:3],lam=lam,blur_level=blur_level)

    if use_tqdm_notebook:
        import tqdm.notebook
        t=tqdm.notebook.trange(niter)
    else:
        t=range(niter)
    for i in t:
        m.update_F(Xsh)
        m.update_alpha(Xsh)
        m.update_a(Xsh)
        m.update_b(Xsh)

    rez=m.F_scaled()
    rez=rez/rez.max()

    return DensityResult(density=rez,model=m,X=Xsh.numpy())
