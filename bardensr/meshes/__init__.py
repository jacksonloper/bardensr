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
from .. import misc


import scipy.spatial


def alpha_3d_shape(pointcloud,circumradius):
    dl = sp.spatial.Delaunay(pointcloud)
    s=dl.simplices
    s=pointcloud[s]
    circumcenters,sizes=misc.calc_circum(s)

    # find out which simplices are good
    good = sizes < circumradius

    # get the good simplices
    good_simplices=dl.simplices[good]

    # turn them into faces
    faces = np.stack([
        good_simplices[:,[0,1,2]],
        good_simplices[:,[0,3,1]],
        good_simplices[:,[0,2,3]],
        good_simplices[:,[1,3,2]],
    ],axis=1)

    # return mesh
    return trimesh.Trimesh(pointcloud,faces.reshape((-1,3)))


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


def voxelize_meshlist_interiors(submeshes,pitch = 100, num_workers=1,
                            use_tqdm_notebook=True):
    '''
    sample interiors of many meshes

    input:
        meshes: a list of trimesh.Trimesh object
        MX,MN : region to voxelize in
        pitch : size of a voxel in resulting dataset
    output:
        rolonies: a list of locations, one for each mesh
    '''
    rolonies=[]
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        t=range(len(submeshes))
        if use_tqdm_notebook:
            t=tqdm.notebook.tqdm(t)
        jobs=[ex.submit(exhaustively_sample_mesh_interior,mesh, pitch) for mesh in submeshes]
        for idx in t:
            rolonies.append(jobs[idx].result())
    return rolonies
