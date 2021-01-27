import h5py
import dataclasses
import numpy as np
import pandas as pd
import scipy as sp
import scipy.spatial

def _locs_and_j_to_df(locs,j):
    return pd.DataFrame(dict(
        m0=locs[:,0],
        m1=locs[:,1],
        m2=locs[:,2],
        j=j
    ))

@dataclasses.dataclass
class RolonyFPFNResult:
    fn:int
    fp:int
    fn_indices:np.array
    fp_indices:np.array
    fn_rolonies:pd.DataFrame
    fp_rolonies:pd.DataFrame
    agreement_rolonies:pd.DataFrame


@dataclasses.dataclass
class BarcodeFPFNResult:
    fn:int
    fp:int
    fdr:float
    dr:float
    barcode_pairing:pd.DataFrame

def codebook_comparison(codebook,other_codebook,tolerated_error=0,strict=False):
    '''
    Attempt to match each code in codebook with a code in other_codebook,
    up to a tolerated error level.

    If strict=False and the entries of a barcode has nans in it,
    we don't consider disagreements there to be errors (just missingness).
    '''

    R,C,J=codebook.shape


    diffs = codebook[:,:,:,None]!=other_codebook[:,:,None,:]
    if not strict:
        diffs[np.isnan(diffs)]=0
    dsts = np.sum(diffs,axis=(0,1)) # J1 x J2

    fp=np.sum(np.min(dsts,axis=0)>tolerated_error) # FP = none of our barcodes are within tolerated error of theirs
    fn=np.sum(np.min(dsts,axis=1)>tolerated_error) # FN = none of their barcodes are within tolerated error of ours

    fdr=fp/other_codebook.shape[-1]
    dr=1.0 - (fn/codebook.shape[-1])

    # get a pairing -- for each of entries in the codebook, find (at least one!) entry in other_codebook
    idx1=[]
    idx2=[]
    for i in range(dsts.shape[0]):
        best=np.argmin(dsts[i])
        if dsts[i,best]<=tolerated_error:
            idx1.append(i)
            idx2.append(best)

    barcode_pairing=pd.DataFrame(dict(
        idx1=idx1,
        idx2=idx2
    ))

    return BarcodeFPFNResult(fn,fp,fdr,dr,barcode_pairing)


def downsample1(x,ds,axis=0):
    newshape=np.array(x.shape,dtype=np.int)
    newshape[axis]=np.ceil(newshape[axis]/ds)
    newguy=np.zeros(newshape,dtype=x.dtype)

    for i in range(ds):
        sl=[slice(0,None) for i in range(len(x.shape))]
        sl[axis]=slice(i,None,ds)
        sl=tuple(sl)
        subx=x[sl]

        sl2=tuple([slice(0,s) for s in subx.shape])
        newguy[sl2]+=subx
    return newguy/ds

def downsample_nd(x,ds):
    if isinstance(ds,int):
        ds=np.full(len(x.shape),ds)
    for i in range(len(x.shape)):
        x=downsample1(x,ds[i],axis=i)
    return x

@dataclasses.dataclass
class Benchmark:
    description: str
    name: str
    version: int
    X: np.array
    codebook: np.array
    rolonies: pd.DataFrame

    def __post_init__(self):
        self.n_spots=len(self.rolonies)
        self.n_genes=self.codebook.shape[-1]
        self.n_good_spots=np.sum(self.rolonies['status']=='good')

    def downsample(self,dsd):
        R,C=self.codebook.shape[:2]
        bc3=self.copy()
        bc3.X=np.array([[downsample_nd(bc3.X[r,c],dsd) for c in range(C)] for r in range(R)])
        bc3.rolonies['m0']=np.require(bc3.rolonies['m0']//dsd,dtype=np.int)
        bc3.rolonies['m1']=np.require(bc3.rolonies['m1']//dsd,dtype=np.int)
        bc3.rolonies['m2']=np.require(bc3.rolonies['m2']//dsd,dtype=np.int)
        return bc3

    def copy(self,copy_imagestack=False,copy_codebook=False):
        if copy_imagestack:
            X=self.X.copy()
        else:
            X=self.X

        if copy_codebook:
            codebook=self.codebook.copy()
        else:
            codebook=self.codebook

        return Benchmark(
            self.description,
            self.name,
            self.version,
            X,
            codebook,
            self.rolonies.copy()
        )

    def save_hdf5(self,fn):
        with h5py.File(fn,'w') as f:
            for nm in ['description','name','version']:
                f.attrs[nm]=getattr(self,nm)
            f.create_dataset('X',data=self.X)
            f.create_dataset('codebook',data=self.codebook)
            f.create_group('rolonies')
            for nm in ['j','m0','m1','m2']:
                ds=np.array(self.rolonies[nm]).astype(np.int)
                f.create_dataset('rolonies/'+nm,data=ds)
            for nm in ['remarks','status']:
                ds=np.array(self.rolonies[nm]).astype("S")
                f.create_dataset('rolonies/'+nm,data=ds)

    def create_new_benchmark_with_more_rolonies(self,df):
        df=df.copy()
        if 'remarks' not in df:
            df['remarks']=np.zeros(len(df),dtype='U80')
        if 'status' not in df:
            df['status']=np.full(len(df),'good',dtype='U80')
        df['status'].fillna(value='good',inplace=True)

        rolonies=pd.concat([self.rolonies,df],ignore_index=True)
        return Benchmark(
            self.description,
            self.name,
            self.version,
            self.X,
            self.codebook,
            rolonies
        )



    def rolony_fpfn(self,df,radius,good_subset=None):
        if len(df)==0:
            noro=_locs_and_j_to_df(
                np.zeros((0,3),dtype=np.int),
                np.zeros(0,dtype=np.int),
            ),
            return RolonyFPFNResult(
                fn=self.n_spots,
                fp=0,
                fn_indices=np.zeros(0,dtype=np.int),
                fp_indices=np.zeros(0,dtype=np.int),
                fp_rolonies=noro,
                fn_rolonies=self.rolonies.copy(),
                agreement_rolonies=noro,
            )

        my_locs=np.c_[
            self.rolonies['m0'],
            self.rolonies['m1'],
            self.rolonies['m2']
        ]
        my_j=np.array(self.rolonies['j'])

        their_locs=np.c_[
            df['m0'],
            df['m1'],
            df['m2'],
        ]
        their_j=np.array(df['j'])

        dsts=sp.spatial.distance.cdist(my_locs,their_locs)

        agreement=(my_j[:,None]==their_j[None,:])
        dsts[~agreement]=np.inf # if the genes don't agree, it doesn't count
        dsts[self.rolonies['status']=='bad']=np.inf # these don't count


        # of the spots that we have that are good
        # how many are in df?
        goodies=self.rolonies['status']=='good'
        if good_subset is not None:
            goodies=goodies&good_subset
        good_locs=my_locs[goodies]
        good_j=my_j[goodies]
        dists_from_goods_to_closest_in_them = np.min(dsts[goodies],axis=1)
        missing_goodies=dists_from_goods_to_closest_in_them>radius
        spots_they_missed=np.sum(missing_goodies)

        # of the spots that they have
        # how many spots do we have?
        dists_from_them_to_closest_in_me_that_isnt_bad = np.min(dsts,axis=0)
        fantasized_bad=dists_from_them_to_closest_in_me_that_isnt_bad>radius
        spots_they_made_up=np.sum(fantasized_bad)

        return RolonyFPFNResult(
            fn=spots_they_missed,
            fp=spots_they_made_up,
            fn_indices=np.where(goodies)[0][missing_goodies], # fn_indices[3] says which benchmark spot we failed at
            fp_indices=np.where(fantasized_bad)[0],
            fn_rolonies=_locs_and_j_to_df(
                good_locs[missing_goodies],
                good_j[missing_goodies]
            ),
            fp_rolonies=_locs_and_j_to_df(
                their_locs[fantasized_bad],
                their_j[fantasized_bad],
            ),
            agreement_rolonies = _locs_and_j_to_df(
                their_locs[~fantasized_bad],
                their_j[~fantasized_bad],
            )
        )


    def __repr__(self):
        return 'Benchmark(description="'+self.description+'",...)'

def query_onehot_codebook(codebook,s):
    assert len(s)==codebook.shape[0]
    good=np.ones(codebook.shape[-1],dtype=np.bool)
    for i,c in enumerate(s):
        if c=='?':
            pass
        else:
            c=int(c)
            good=good&codebook[i,c]
    return np.where(good)[0]

def load_h5py(fn):
    dct={}
    with h5py.File(fn,'r') as f:
        for nm in ['description','name','version']:
            dct[nm]=f.attrs[nm]
        dct['X']=f['X'][:]
        dct['codebook']=f['codebook'][:]

        rn={}
        for nm in ['j','m0','m1','m2']:
            rn[nm]=f['rolonies/'+nm][:].astype(np.int)
        for nm in ['remarks','status']:
            rn[nm]=f['rolonies/'+nm][:].astype('U')
        dct['rolonies']=pd.DataFrame(rn)
    bc=Benchmark(**dct)
    bc.source_fn=fn
    return bc
