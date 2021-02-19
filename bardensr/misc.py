import numpy as np

def argmax_nd(x):
    return np.unravel_index(np.argmax(x),x.shape)


def calc_circum(V):
    '''
    input
    * V -- (batch x (d+1) x d) -- a simplex
    
    Output
    * circumcenters -- (batch,d)
    * circumradii -- (batch)
    '''
    
    '''
    Let
    
        diff = V[i,j1] - V[i,j2]
        diff = diff/np.linalg.norm(diff)
    
    Then circumcenter[i] must satisfy
    
        np.sum(circumcenter[i]*diff) == .5*np.sum(diff*(V[i,j1]+V[i,j2]))
    
    This forms a system we can work with to compute circumcenters
    and circumradii.
    '''
    
    directions = V[:,[0]] - V[:,1:]  # batch x d x d
    avgs = .5*(V[:,[0]] + V[:,1:]) # batch x d x d
    directions = directions / np.linalg.norm(directions,keepdims=True,axis=-1) # batch x d x d
    
    beta = np.sum(avgs*directions,axis=-1)
    
    circumcenters = np.linalg.solve(directions,beta) # batch x d
    circumradii = np.linalg.norm(circumcenters - V[:,0],axis=-1)
    
    return circumcenters,circumradii
    

def nan_robust_hamming(A,B):
    '''
    Input:
    - A (M x N1 )
    - B (M x N2 )

    Output
    - diffs (N1 x N2)

    diffs[i,j] = #{k: both A[i,k],B[i,k] are non-nan and A[i,k]!=B[i,k}
    '''

    M,N1=A.shape
    M,N2=A.shape

    differences=0

    for m in range(M):
        subdifferences = np.abs(A[m,:,None] - B[m,None,:]).astype(float) # N1 x N2

        # if one of the barcodes says it doesn't know about one of the frames
        # then we say that barcode doesnt disagree about that fram
        subdifferences[np.isnan(subdifferences)]=0

        # compute the total number of disagreements for each pair of barcodes
        differences+=subdifferences # N1 x N2

    return differences

def match_onehot(codebook,s):
    good=np.ones(codebook.shape[-1],dtype=np.bool)
    for r,c in enumerate(s):
        if c=='?':
            pass
        else:
            c=int(c)
            good=good&codebook[r,c,:]
    return np.where(good)[-1]

def maybe_trange(n,use_tqdm_notebook):
    if use_tqdm_notebook:
        import tqdm.notebook
        return tqdm.notebook.trange(n)
    else:
        return range(n)

def maybe_tqdm(n,use_tqdm_notebook,*args,**kwargs):
    if use_tqdm_notebook:
        import tqdm.notebook
        return tqdm.notebook.tqdm(n,*args,**kwargs)
    else:
        return n

def convert_codebook_to_onehot_form(codebook):
    '''
    Input: codebook, JxR
    Output: codebook, RxCxJ
    '''
    J,R=codebook.shape
    C=codebook.max()+1
    codes=np.eye(C,dtype=np.float)
    codes=np.concatenate([codes,np.full((1,C),np.nan)],axis=0)
    codebook=codes[codebook.ravel()].reshape((J,R,C))
    return np.transpose(codebook,[1,2,0])
