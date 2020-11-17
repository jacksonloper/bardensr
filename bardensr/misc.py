import numpy as np

def argmax_nd(x):
    return np.unravel_index(np.argmax(x),x.shape)

def match(codebook,s):
    good=np.ones(codebook.shape[-1],dtype=np.bool)
    for r,c in enumerate(s):
        if c=='?':
            pass
        else:
            c=int(c)
            good=good&codebook[r,c,:]
    return np.where(good)[-1]
