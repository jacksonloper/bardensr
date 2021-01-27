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

def convert_codebook_to_onehot_form(codebook):
    '''
    Input: codebook, JxR
    Output: codebook, RxCxJ
    '''
    J,R=codebook.shape
    C=codebook.max()+1
    codes=np.eye(C,dtype=np.bool)
    codebook=codes[codebook.ravel()].reshape((J,R,C))
    return np.transpose(codebook,[1,2,0])
