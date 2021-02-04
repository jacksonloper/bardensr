import tqdm, sys
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import scipy as sp
import pickle
import trimesh
from concurrent.futures import ProcessPoolExecutor
import bardensr.misc
import tqdm.notebook

def dilate_fill(a):
    b = np.zeros_like(a)
    for m in range(a.shape[0]):
        b[m] = sp.ndimage.binary_fill_holes(a[m])
    return b

def voxelize(mesh, pitch, mins,maxs):
    '''
    input:
        mash: watertight mesh
        pitch: downsample level
        maxs/mins: boundaries of voxelation; must divide evenly by pitch
    output:
        out: voxelized interior of mesh

    Specifically, out[i,j,k] is True if the voxel spanning the cube

        mins[0]+pitch*i -- mins[0]*pitch+(i+1)
        mins[1]+pitch*j -- mins[1]*pitch+(j+1)
        mins[2]+pitch*k -- mins[2]*pitch+(k+1)

    is belived to intersect with the interior of the mesh.

    '''
    vox=get_boundary_voxels(mesh,pitch,mins,maxs)
    return dilate_fill(vox)

def get_boundary_voxels(mesh:trimesh.Trimesh, pitch,mins,maxs):
    '''
    input:
        mash: original mesh
        pitch: downsample level
        maxs/mins: boundaries of voxelation; must divide evenly by pitch
    output:
        vox: voxelized version of exterior of mesh

    Specifically, out[i,j,k] is True if the voxel spanning the cube

        mins[0]+pitch*i -- mins[0]*pitch+(i+1)
        mins[1]+pitch*j -- mins[1]*pitch+(j+1)
        mins[2]+pitch*k -- mins[2]*pitch+(k+1)

    is belived to intersect with the mesh surface.
    '''

    assert (maxs>mins).all()
    assert np.allclose(mins%pitch,0)
    assert np.allclose(mins%pitch,0)

    maxs=np.array(maxs)
    mins=np.array(mins)
    pitch=float(pitch)

    # out will be this size
    out= np.zeros(np.require((maxs-mins)//pitch,dtype=np.int),dtype=np.bool)

    # get hits, at pitch-level resolution
    max_edge = pitch / 2.0
    v,f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge)

    # convert hit so that one unit=one pitch
    hit = (v / pitch)

    # integerify and uniquify
    hit = hit.astype(np.int)
    hit = np.unique(hit,axis=0)

    # convert mins,maxs so that one unit = one pitch
    mins=np.require(np.floor(mins/pitch),dtype=np.int)
    maxs=np.require(np.ceil(maxs/pitch),dtype=np.int)

    # figure out how big the output array ought to be
    size=maxs-mins

    # kill those we don't care about
    for i in range(3):
        hit=hit[hit[:,i]>=mins[i]]
    for i in range(3):
        hit=hit[hit[:,i]<maxs[i]]

    # translate so 0=mins
    hit=hit-mins

    # make it!
    out=np.zeros(size,dtype=np.bool)
    out[hit[:,0],hit[:,1],hit[:,2]]=True

    return out

def sample_mesh_interior(mesh, poisrate, pitch):
    '''
    for one mesh, vozelize and return the rolony based on the poisson rate.
    input:
        mesh: watertight trimesh.Trimesh object
        poisrate: density/(unit volume)
        pitch: approximation level to use in sampling (lengthscale used for voxelation)
    output:
        locations: simulated spots inside the mesh; these locations will be divisible by pitch
    '''
    MX=pitch*(1+np.max(mesh.vertices,axis=0)//pitch)
    MN=pitch*(np.min(mesh.vertices,axis=0)//pitch)
    vv =voxelize(mesh,pitch,maxs=MX,mins=MN)
    opts=np.array(np.where(vv)).T
    counts=np.random.poisson(poisrate*pitch*pitch*pitch,size=len(opts))
    opts=opts*pitch+np.array(MN)
    finalopts=np.repeat(opts,counts,axis=0)
    return finalopts

def exhaustively_sample_mesh_interior(mesh,pitch):
    '''
    for one mesh, vozelize and return the rolony based on the poisson rate.
    input:
        mesh: watertight trimesh.Trimesh object
        pitch: approximation level to use in sampling (lengthscale used for voxelation)
    output:
        locations: all voxel locations insite the mesh
    '''
    MX=pitch*(1+np.max(mesh.vertices,axis=0)//pitch)
    MN=pitch*(np.min(mesh.vertices,axis=0)//pitch)
    vv =voxelize(mesh,pitch,maxs=MX,mins=MN)
    opts=np.array(np.where(vv)).T
    opts=opts*pitch+np.array(MN)
    return opts


def sample_meshlist_interiors(submeshes,pitch = 100, poisrate=1e-8,num_workers=1,
                            use_tqdm_notebook=True):
    '''
    sample interiors of many meshes

    input:
        meshes: a list of trimesh.Trimesh object
        MX,MN : region to voxelize in
        pitch : size of a voxel in resulting dataset
        poisrate: rate of poisson - higher, more rolonies are simulated
    output:
        rolonies: a list of locations, one for each mesh
    '''
    rolonies=[]
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        t=range(len(submeshes))
        if use_tqdm_notebook:
            t=tqdm.notebook.tqdm(t)
        if poisrate==np.inf:
            jobs=[ex.submit(exhaustively_sample_mesh_interior,mesh, pitch) for mesh in submeshes]
        else:
            jobs=[ex.submit(sample_mesh_interior,mesh,poisrate, pitch) for mesh in submeshes]
        for idx in t:
            rolonies.append(jobs[idx].result())
    return rolonies

def rol2X(rolonies,
          blursz = 2,
          noiselevel = .01,
          m2range = (0,25), R = 17, C = 4
         ):
    '''
    create the benchmark dataset from the rolonies pd dataframe
    input:
        rolonies: the returned object from mesh2rol, pandas dataframe
        m2thre: the range to look at (from 0 to m2thre)
    output:
        X: size of (R, C, M0, M1, M2)
    '''
    # restrict to m2<m2thre
    good=(rolonies['m2']>=m2range[0]) & (rolonies['m2']<m2range[1])  # what is this 25??
    rolonies=rolonies[good]
    unq,rev=np.unique(rolonies['j'],return_inverse=True)
#     rolonies['j']=rev   # what for ?
    J=rolonies['j'].max()+1
    codebook=np.random.randint(0,C,size=(J,R))
    codebook=bardensr.misc.convert_codebook_to_onehot_form(codebook)
    shape=(rolonies['m0'].max()+1,rolonies['m1'].max()+1,rolonies['m2'].max()+1)
    X=np.zeros((R, C)+shape)
    for j in tqdm.tqdm_notebook(range(J)):
        subX=np.zeros_like(X)
        good=rolonies['j']==j
        subr=rolonies[good]
        scales=np.random.rand(len(subr))+1
        for r,c in zip(*np.where(codebook[:,:,j])):
            subX[r,c,subr['m0'],subr['m1'],subr['m2']]+=scales
        X+=subX
    X=sp.ndimage.gaussian_filter(X,(0,0,blursz, blursz, blursz))
    X=X+np.random.randn(*X.shape)*X.max()*noiselevel
    X=np.clip(X,0,None)
    return(X, codebook)






# def compute_smoothdorff(u, v, max_dis = 100):
#     '''
#     compute smoothdorff between u and v
#     input:
#         u/v: both numpy array (S, 2).  (TODO: 3d array?)
#         Note v should be GT and cannot be empty.
#         u can be empty in which case return the max_dis.
#     output: a scalar. smaller, closer the predicted to the GT is.
#     '''
#     assert(v.shape[0] > 0)
#     if (u.shape[0] == 0):
#         return(max_dis)  # empty set for the detected spots --> bad!
#     else:
#         kdtree_u  = sp.spatial.KDTree(u.astype(float))
#         kdtree_v  = sp.spatial.KDTree(v.astype(float))
#         sdm = kdtree_u.sparse_distance_matrix(kdtree_v, max_dis)
#         sdm_csr = sdm.tocsr()  # S_v x S_u in dense
#         min0 = sdm_csr.min(axis = 0).astype(float)
#         min1 = sdm_csr.min(axis = 1).astype(float)
#         out = max(min0.todense().mean(), min1.todense().mean())
#         return(out)
