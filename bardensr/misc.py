__all__ = [
    'argmax_nd',
]

import numpy as np
import tempfile
import zipfile
import io
import os
import shutil
import tensorflow as tf

def argmax_nd(x):
    '''
    Compute the argmax of a tensor.

    - Input: x, an n-dimensional numpy array
    - Output: idx: an integer n-tuple indicating an index
      where x reaches its greatest point

    Seems like this is the kind of think numpy might provide,
    but I think it isn't.
    '''
    return np.unravel_index(np.argmax(x),x.shape)

def tf_model_to_wire(model):
    # save to the wire
    with tempfile.TemporaryDirectory() as tempdirname:
        tf.saved_model.save(model,tempdirname)
        shutil.make_archive(tempdirname+"/packed",'zip',tempdirname)
        with open(tempdirname+"/packed.zip",'rb') as f:
            packed_zipfile_string=f.read()
    return packed_zipfile_string

def tf_model_from_wire(s):
    # get out of the wire
    with tempfile.TemporaryDirectory() as tempdirname,io.BytesIO(s) as packed_zipfile_io:
        with zipfile.ZipFile(packed_zipfile_io) as zf:
            zf.extractall(tempdirname)
        model=tf.saved_model.load(tempdirname)

    return model



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

def maybe_tqdm_ray(jobs,use_tqdm_notebook):
    import ray
    if use_tqdm_notebook:
        import tqdm.notebook
        def toit(job_ids):
            while job_ids:
                done,job_ids =ray.wait(job_ids)
                yield ray.get(done[0])
        return tqdm.notebook.tqdm(toit(jobs),total=len(jobs))
    else:
        return [ray.get(x) for x in jobs]


def convert_codebook_to_onehot_form(codebook,C=None):
    '''
    Input: codebook, JxR
    Output: codebook, RxCxJ
    '''
    J,R=codebook.shape
    if C == None:
        C=codebook.max()+1
    codes=np.eye(C,dtype=np.float)
    codes=np.concatenate([codes,np.full((1,C),np.nan)],axis=0)
    codebook=codes[codebook.ravel()].reshape((J,R,C))
    return np.transpose(codebook,[1,2,0])

def ray_batch(f,arglist,use_tqdm_notebook=False):
    try:
        import ray
    except ModuleNotFoundError:
        raise ModuleNotFoundError("batching methods require the ray package") from None

    assert ray.is_initialized(),'user should initialize ray with "import ray; ray.init()" before calling register_batched'

    if use_gpus:
        @ray.remote(num_gpus=1)
        def go(arg):
            ids=ray.get_gpu_ids()
            assert len(ids)==1
            gpuid=ids[0]
            with tf.device(f"gpu:{gpuid}"):
                return f(arg)
    else:
        go = ray.remote(f)

    jobs=[go.remote(arg) for arg in arglist]
    return misc.maybe_tqdm_ray(jobs, use_tqdm_notebook)
