


import numpy as np
import scipy as sp
import scipy.spatial.distance
from .. import kernels
import numpy.random as npr

def sim_onehot_code_against_unknown_codes(codes,lim=100,hamdst=3):
    R,C,J=codes.shape
    N=R*C

    codes = np.reshape(codes,(N,J)).T

    for i in range(lim):
        newhot = npr.randint(0,C,size=R)
        newcode=np.zeros((R,C))
        newcode[np.r_[0:R],newhot]=1
        if sp.spatial.distance.cdist(newcode.ravel()[None,:],codes).min()<hamdst:
            return newcode
    raise Exception("Unable to find another barcode")

def sim_onehot_code(R,C,hots,lim=100,hamdst=3):
    hots=np.array(hots)
    for i in range(lim):
        newhot = npr.randint(0,C,size=R)
        if sp.spatial.distance.cdist(newhot[None,:],hots).min()<hamdst:
            return newhot
    raise Exception("Unable to find another barcode")

def sim_onehot_codes(R,C,J,lim=100,hamdst=3):
    hots=[npr.randint(0,C,size=R)]
    for j in range(1,J):
        hots.append(sim_onehot_code(R,C,hots,lim=lim,hamdst=hamdst))
    return hots

def unif(a,b,*shape):
    return npr.rand(*shape)*(b-a)+a
def simulate(
        spatial_dims=(200,210,5),
        rolony_size=3,
        n_barcodes=81,
        n_rounds=7,
        n_channels=4,
        codebook='one-hot',  # <-- simulate a one-hot codebook
        genedistr=None, # <-- meaning uniform
        n_rolonies=400,
        required_hamdst=3,
        noise=.01,
        scale_lo=.5,
        scale_hi=1.5,
):
    '''
    Simulates an observation tensor according to the singlefov model (described above).

    Input:
    - spatial_dims -- m1,m2,m3
    - rolony_size -- magnitude of the blur kernel K
    - n_rounds -- R
    - n_channels -- C
    - codebook -- see below
    - genedistr -- when creating F, the number of nonzero entries in F
    will be distributed so that $\sum_{m_1,m_2,m_3} F_{m_1,m_2,m_3,j} \propto \mathtt{genedistr}_j$
    - n_rolonies -- number of nonzero entries in F
    - required_hamdst -- see below
    - noise -- magnitude of noise which causes observations to deviate from model
    - scale_lo -- lowest magnitude of signal in F
    - scale_hi -- highest magnitude of signal in F

    Output: a dictionary with
    - data: a M1 x M2 x M3 x R x C tensor
    - ground_truth_densities: sparse representation of the nonzero entries in F
    - codebook: the codebook B used to generate data

    a word on the codebook.  this parameter can have one of two types:
    1. if codebook == 'one-hot' it will make up a codebook with the right
    number of rounds and channels.  it will require that all the codes are
    at least required_hamdst different from each other, and it will assume
    that the codebook is binary and (np.sum(codebook,1)==1).all().
    2. if codebook is an array, R,C,required_hamdst will be ignored, and
    we will just use the given codebook

    '''

    # make codebook
    if codebook=='one-hot':
        R=n_rounds
        C=n_channels
        J=n_barcodes

        hots = sim_onehot_codes(R,C,J,lim=100,hamdst=required_hamdst)
        codebook=np.zeros((R,C,J),dtype=np.float64)
        for j in range(J):
            codebook[np.r_[0:R],hots[j],j]=1
    else:
        try:
            codebook=np.require(codebook,dtype=np.float64)
            assert len(codebook.shape)==3
            R,C,J=codebook.shape
        except Exception:
            raise Exception("Unable to interpret codebook")

    # format spatial dims
    spatial_dims=tuple(spatial_dims)
    M=np.prod(spatial_dims)

    # get rolony densities
    if genedistr is None:
        genedistr=np.ones(J,dtype=np.float64)/J

    # get positions
    positions = npr.randint(0,M,size=n_rolonies)
    genes = npr.choice(J,p=genedistr,size=n_rolonies)  # genes[i] ~ Categorical(genedistr)
    F=np.zeros((M,J))
    vals=unif(scale_lo,scale_hi,n_rolonies)
    F[positions,genes]=vals
    ground_truth=np.where(F>0)
    vals=F[ground_truth]

    # blur 'em
    F=F.reshape(spatial_dims+(J,))
    F=kernels.heat_kernel_nd(F,(rolony_size,)*len(spatial_dims))
    F=np.reshape(F,(M,J))

    # make observations
    X=np.einsum('mj,rcj->mrc',F,codebook)

    # noise them up
    X=X+npr.rand(M,R,C)*noise

    # return a simulation
    pos2 = np.c_[np.unravel_index(ground_truth[0],spatial_dims)]
    gene2 = ground_truth[1]

    return dict(
        data=X.reshape(spatial_dims+(R,C)),
        ground_truth_positions=dict(m1=pos2[:,0],m2=pos2[:,1],m3=pos2[:,2],j=gene2,values=vals),
        codebook=codebook,
    )
