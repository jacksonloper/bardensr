import h5py
import dataclasses
import numpy as np
import collections
import pandas as pd
import scipy as sp
import scipy.spatial
from typing import Optional
import tqdm
import trimesh
from .. import misc
from . import simulations
from . import locsdf

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
class BarcodePairing:
    pairing_list:np.array

    def __post_init__(self):
        self.book1_lookup=collections.defaultdict(set)
        self.book2_lookup=collections.defaultdict(set)

        if len(self.pairing_list)>0:

            for i,j in self.pairing_list:
                self.book1_lookup[i].add(j)
                self.book2_lookup[j].add(i)

            self.book1_maps_unambiguously = (np.max([len(self.book1_lookup[j]) for j in self.book1_lookup])<=1)
            self.book2_maps_unambiguously = (np.max([len(self.book2_lookup[j]) for j in self.book2_lookup])<=1)
        else:
            self.book1_maps_unambiguously = True
            self.book2_maps_unambiguously = True


@dataclasses.dataclass
class BarcodeFPFNResult:
    fn:int
    fp:int
    fdr:float
    dr:float
    barcode_pairing:BarcodePairing
    true_detects:int

    def __repr__(self):
        return f'[barcode comparison: false positives={self.fp}, dr={self.dr*100:04.1f} ({self.true_detects} detects)%]'

def codebook_comparison(codebook,other_codebook,tolerated_error=0,strict=False):
    '''
    Attempt to match each code in codebook with a code in other_codebook,
    up to a tolerated error level.

    If strict=False and the entries of a barcode has nans in it,
    we don't consider disagreements there to be errors (just missingness).
    '''

    R,C,J=codebook.shape


    codebook=codebook.astype(np.float)
    other_codebook=other_codebook.astype(np.float)
    dsts = misc.nan_robust_hamming(codebook.reshape((R*C,-1)),other_codebook.reshape((R*C,-1)))

    fp=np.sum(np.min(dsts,axis=0)>tolerated_error) # FP = none of our barcodes are within tolerated error of theirs
    fn=np.sum(np.min(dsts,axis=1)>tolerated_error) # FN = none of their barcodes are within tolerated error of ours

    fdr=fp/other_codebook.shape[-1]
    dr=1.0 - (fn/codebook.shape[-1])

    true_detects = codebook.shape[-1] - fn

    idx1,idx2=np.where(dsts<=tolerated_error)

    barcode_pairing=BarcodePairing(np.c_[idx1,idx2])

    return BarcodeFPFNResult(fn,fp,fdr,dr,barcode_pairing,true_detects)

def meanmin_divergence(u,v):
    '''
    Input
    - u, (NxD) matrix
    - v, (MxD) matrix
    Output:
        meanmin(u<v) = mean_i min_j |u[i]-v[j]|
    If u is a subset of v, meanmin=0
    Is u is nontrivial and v is empty, minmin = inf
    '''

    if len(u)==0: # then, by definition, l1 is a subset of l2
        return 0.0
    elif len(v)==0: # there's stuff in l1, but NOTHING in l2
        return np.inf
    else:
        import sklearn.neighbors
        X=sklearn.neighbors.BallTree(v)
        dists=X.query(u,k=1,return_distance=True)[0]
        return np.mean(dists)


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
    GT_voxels: Optional[list] = None  # list of length J.
    GT_meshes: Optional[list] = None  # list of (vertices,faces) of length J
    units: Optional[str] = None # information about voxel units,

    def __post_init__(self):
        self.n_spots=len(self.rolonies)
        self.n_genes=self.codebook.shape[-1]
        if 'status' not in self.rolonies:
            self.rolonies['status']=np.full(len(self.rolonies),'good',dtype='U40')
        if 'remarks' not in self.rolonies:
            self.rolonies['remarks']=np.full(len(self.rolonies),'',dtype='U40')
        self.n_good_spots=np.sum(self.rolonies['status']=='good')
        if type(self.GT_voxels) == type(None):
            self.GT_voxels = pd.DataFrame(data = [], columns = ['j','m0','m1','m2'])

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
            for nm in ['description','name','version','units']:
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

            if self.GT_voxels is not None:
                f.create_group('GT_voxels')
                for nm in ['j','m0','m1','m2']:
                    ds=np.array(self.GT_voxels[nm]).astype(np.int)
                    f.create_dataset('GT_voxels/'+nm,data=ds)

            if self.GT_meshes is not None:
                f.create_group('GT_meshes')
                self.v_list = [mesh.vertices for mesh in self.GT_meshes]
                self.f_list = [mesh.faces for mesh in self.GT_meshes]
                for i, (vertices,faces) in enumerate(zip(self.v_list, self.f_list)):
                    f.create_dataset('GT_meshes/'+str(i)+'/vertices', data = vertices)
                    f.create_dataset('GT_meshes/'+str(i)+'/faces', data = faces)

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

    def voxel_meanmin_divergences(self,df,barcode_pairing=None,use_tqdm_notebook=False):
        '''
        Input:
        - df, a dataframe of rolonies
        - [optional] barcode_pairing, matching (js from self.rolonies) <--> (js from df)

        Output:
        - us_c_them_errors -- for each j, the failure of our voxels to be a subset of their voxels
        - them_c_us_errors -- for each j, the failure of their voxels to be a subset of our voxels
        - unmatched_barcodes -- for each j, whether that barcode was simply absent from the barcode pairing
        '''

        us_c_them_errors=np.zeros(self.n_genes)
        them_c_us_errors=np.zeros(self.n_genes)

        if barcode_pairing is not None:
            assert barcode_pairing.book2_maps_unambiguously,"some of the df barcodes are mapped to more than one of our barcodes!"

        for j in misc.maybe_trange(self.n_genes,use_tqdm_notebook):
            l1=self.GT_voxels[self.GT_voxels['j']==j][['m0','m1','m2']]

            if barcode_pairing is not None:
                l2=df[df['j'].isin(barcode_pairing.book1_lookup[j])][['m0','m1','m2']]
            else:
                l2=df[df['j']==j][['m0','m1','m2']]

            us_c_them_errors[j]=meanmin_divergence(l1,l2) # if l1 is subset of l2 -- >, l1 fails to cover l2
            them_c_us_errors[j]=meanmin_divergence(l2,l1)

        unmatched_barcodes=np.zeros(self.n_genes,dtype=np.bool)
        if barcode_pairing is not None:
            for j in range(self.n_genes):
                if len(barcode_pairing.book1_lookup[j])==0:
                    unmatched_barcodes[j]=True

        return us_c_them_errors,them_c_us_errors,unmatched_barcodes

    def rolony_fpfn(self,df,radius,good_subset=None):
        if len(df)==0:
            noro=locsdf.locs_and_j_to_df(
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
            fn_rolonies=locsdf.locs_and_j_to_df(
                good_locs[missing_goodies],
                good_j[missing_goodies]
            ),
            fp_rolonies=locsdf.locs_and_j_to_df(
                their_locs[fantasized_bad],
                their_j[fantasized_bad],
            ),
            agreement_rolonies=locsdf.locs_and_j_to_df(
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
        for nm in ['description','name','version','units']:
            dct[nm]=f.attrs[nm]
        dct['X']=f['X'][:]
        dct['codebook']=f['codebook'][:]

        rn={}
        for nm in ['j','m0','m1','m2']:
            rn[nm]=f['rolonies/'+nm][:].astype(np.int)
        for nm in ['remarks','status']:
            rn[nm]=f['rolonies/'+nm][:].astype('U')
        dct['rolonies']=pd.DataFrame(rn)

        if 'GT_voxels' in f:
            rn={}
            for nm in ['j','m0','m1','m2']:
                rn[nm]=f['GT_voxels/'+nm][:].astype(np.int)
            dct['GT_voxels']=pd.DataFrame(rn)
        else:
            dct['GT_voxels']=None

        if 'GT_voxels' in f:
            mesh_list = []
            for i in range(len(f['GT_meshes'])):
                vertices =  f['GT_meshes/'+str(i)+'/vertices']
                faces = f['GT_meshes/'+str(i)+'/faces']
                mesh_list.append(trimesh.Trimesh(vertices, faces, process=False))
            dct['GT_meshes'] = mesh_list
        else:
            dct['GT_voxels']=None


    bc=Benchmark(**dct)
    bc.source_fn=fn
    return bc
