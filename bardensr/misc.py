import numpy as np

def argmax_nd(x):
    return np.unravel_index(np.argmax(x),x.shape)

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
    codes=np.eye(C,dtype=np.bool)
    codes=np.concatenate([codes,np.full((1,C),np.nan)],axis=0)
    codebook=codes[codebook.ravel()].reshape((J,R,C))
    return np.transpose(codebook,[1,2,0])
