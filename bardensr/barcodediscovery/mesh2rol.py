import tqdm, sys
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
import scipy as sp
import pickle
import trimesh
from concurrent.futures import ProcessPoolExecutor
import bardensr.misc


def dilate_fill(a):
    '''
    dilate and fill the holes in a, across axis i. 
    using ndi.binary_fill_holes. 
    axis i is determined to be the smallest axis. 
    '''
    a_dilate = ndi.binary_dilation(a)
    b = np.zeros_like(a)
    i = np.argmin(a.shape)
    n = a.shape[i]
    for m in range(n):
        b[m] = ndi.binary_fill_holes(a_dilate[m])        
    return(b)



def voxelize(mesh, pitch, maxs, mins):
    '''
    input:
        mash: original mesh
        pitch: downsample level 
        maxs/mins: maxs and mins computed across `allguys` - the output will be the same size. 
    output:
        out: 3d array, mesh are converted to the voxels in the array. 
        vox: voxelized mesh object
    '''
    max_edge = pitch / 2.0
    v,f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge)
    hit = (v / pitch)
    hit = np.vstack((np.ceil(hit), np.floor(hit))).astype(int)
    u,i = trimesh.grouping.unique_rows(hit)
    occupied = hit[u]*pitch
    vox = trimesh.voxel.ops.multibox(occupied, pitch)    
    out = np.zeros(np.ceil((maxs-mins)/pitch + 1).astype(int), dtype = 'int')    
    for x,y,z in (mesh.vertices - mins).astype(int):
        out[int(x/pitch),int(y/pitch),int(z/pitch)] = 1
    out = dilate_fill(out)    
    return(out, vox)




def load_n_filter_meshes(fn, nanometers_per_weirdunits=3.58/4, pix_thre = 15_000):    
    '''
    the function that reads pickle file of meshes,
    and filter the meshes based on the length. 
    input:
        fn: location of the pickle file
    output is a dictionary with:
        meshes (list of mesh objects)
        MX, MN: max and min of the vertices for all the meshes. 
    '''
    with open(fn,'rb') as f:
        meshes=[]
        for (v,f) in pickle.load(f):
            v=v*nanometers_per_weirdunits
            meshes.append(trimesh.Trimesh(v,f))
    # convert to nanometers
    THRESH=pix_thre*nanometers_per_weirdunits
    sizes=np.array([np.max(np.ptp(x.vertices,axis=0)) for x in meshes])    
    meshout = dict(meshes = [meshes[x] for x in np.where(sizes>THRESH)[0]], 
                   MX = np.max([x.vertices.max(axis=0) for x in meshes],axis=0), 
                   MN = np.min([x.vertices.min(axis=0) for x in meshes],axis=0), 
                  )
    return(meshout)

def rol_for_one(mesh, MX, MN, poisrate, pitch):
    '''
    for one mesh, vozelize and return the rolony based on the poisson rate. 
    input:
        mesh: one mesh axon object
        MX/MN: max nad min of the entire 3d array size (define the final size of the array)
    output:
        opts: simulated spots locations. 
    '''
    vv, _=voxelize(mesh,pitch,MX,MN)
    opts=np.array(np.where(vv)).T
    counts=np.random.poisson(poisrate*(1e6),size=len(opts))
    opts=opts[counts>0]
    return opts
    
    
def mesh2rol(meshout, pitch = 100, poisrate=1e-8): 
    '''
    simulate spots and create rolony dataframe from the submeshes
    run jobs for individual mesh object, using parallel
    
    input:
        meshout: filtered meshes dictionary from above function
        poisrate: rate of poisson - higher, more rolonies are simulated
    output:
        rolonies: pandas dataframe. 
    '''
    rolonies=[]    
    submeshes=meshout['meshes']#[m for (m,s) in zip(meshes,status) if s!='bad']
    MX = meshout['MX']
    MN = meshout['MN']    
    bins=np.r_[0:len(submeshes):10,len(submeshes)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i in tqdm.tqdm_notebook(range(len(bins)-1)):
            st,en=bins[i],bins[i+1]
            jobs=[ex.submit(rol_for_one,submeshes[i], MX, MN, poisrate, pitch) for i in range(st,en)]
            for i in range(st,en):
                idx=i-st
                vv=jobs[idx].result()
                vv=np.c_[vv,np.full(len(vv),i)]
                rolonies.append(vv)
    rolonies=np.concatenate(rolonies,axis=0)
    rolonies=pd.DataFrame(dict(
        m0=rolonies[:,0],
        m1=rolonies[:,1],
        m2=rolonies[:,2],
        j=rolonies[:,3],
    ))
    return(rolonies)




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