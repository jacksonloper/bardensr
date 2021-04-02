import numpy as np
import numpy.random as npr
from .. import rectangles
import tensorflow as tf

def box_line_intersect(box_st,box_en,loc,angle):
    '''
    Let L = {x: x=loc + angle*a for some scalar a}
    Let box be the set defined as the product of intervals:

        box = [box_st[0],box_en[0]] X [box_st[1],box_en[1]] ...

    Find w1,w2 such that the intersection between L and box
    is equal to the convex hull of {w1,w2}
    '''

    d=len(box_st)
    loc=np.require(loc)
    angle=np.require(angle)
    angle = angle/np.sqrt(np.sum(angle**2))

    # ensure loc is inside box
    if not rectangles.pointcloud_in_box_test(loc[None],box_st,box_en,
                                                      left_closed=False,right_closed=False)[0]:
        raise ValueError("loc must be strictly in the interior of the box")

    # find intersection with each side of the box
    intrscts=[]
    for i in range(d):
        if angle[i]==0:
            pass
        else:
            intrscts.append(loc + ((box_st[i]-loc[i])/angle[i])*angle)
            intrscts[-1][i]=box_st[i]
            intrscts.append(loc + ((box_en[i]-loc[i])/angle[i])*angle)
            intrscts[-1][i]=box_en[i]
    intrscts=np.array(intrscts)

    good=rectangles.pointcloud_in_box_test(intrscts,box_st,box_en,
                                                    left_closed=True,right_closed=True)

    return intrscts[good]

def cylinder_distances(locs,directions,ms,ensure_unit_directions=True):
    '''
    Input:
    - locs: n x J
    - directions: n x J
    - ms: M1 x M2 ... Mn x n
    - ensure_unit_directions: if True, directions[:,j] is normalized for each j
                               if False, user must ensure this is true

    Output is cylinders: M0 x M1 ... Mn x J, binary

    cylinders is defined such that

        cylinders[j,m0,m1...]

    gives the squared distance between the point ms[m0,m1,...] and the line

        L = {x: x=loc[j] + directions[j]*alpha for some alpha}
    '''

    if ensure_unit_directions:
        directions=directions/tf.math.sqrt(tf.reduce_sum(directions**2,axis=0))

    # make cylinder
    df   = locs-ms[...,None]   # e.g  (M0,M1,M2,3,J)
    dsts = tf.reduce_sum(df**2,axis=-2) # e.g. (M0,M1,M2,J)
    proj = tf.reduce_sum(df*directions,axis=-2)**2 # e.g (M0,M1,M2,J)

    return dsts-proj

def simulate_from_length(J,axonradius_vox = 1,  # units in voxel
                          rolplength = 0.08, # rol per micron
                          voxsize = (5,5,20),  # units in micron
                          box = (300,300,300),  # units in micron)
                          loc_centering=.05,
                         ):
    '''
    output
    - GT_voxels (M0,M1,..), binary
    - rolonies (M0,M1,...), binary
    '''

    box=np.require(box).astype(float)
    voxsize=np.array(voxsize)
    nd=len(box)
    assert loc_centering<=.5

    # get cylinder and positions of rolonies along cylinder
    # in units of voxels
    lars=[]
    for j in range(J):
        lars.append=simulate_from_length_sparse(rolplength,box,loc_centering)

    # get meshgrid
    voxel_centers=[np.r_[0:b:v]//v for (b,v) in zip(box,voxsize)] # voxel units
    ms=np.stack(np.meshgrid(*voxel_centers,indexing='ij'),axis=-1).astype(float)  # unit in voxels

    # make cylinder
    df   = loc-ms   #(M0,M1,M2,3)  # unit in voxels.
    dsts = np.sum(df**2,axis=-1) #(M0,M1,M2)
    proj = np.sum(df*angle,axis=-1)**2 #(M0,M1,M2)
    cylinder = np.sqrt(dsts-proj) < axonradius_vox  # voxels comparison.

    # make rolonies
    r=np.zeros(cylinder.shape,dtype=np.bool)
    r[tuple(rolpos.T)]=True

    return cylinder,r

def simulate_from_length_sparse(
                          rolplength = 0.08,
                          box = (300,300,300),
                          loc_centering=.05,
                         ):
    '''
    Simulates a random line and some random points along that line
    that lie within the box.

    output
    - loc, 3-vector (in voxel-space)
    - angle, 3-vector (in voxelspace)
    - rolpos, N x 3 integers (in voxelspace)
    '''

    box=np.array(box).astype(float)
    nd=len(box)
    assert loc_centering<=.5

    # pick a location in the box
    loc = np.random.rand(nd)*box*(1-loc_centering*2)+box*loc_centering

    # pick a direction
    angle = np.random.randn(nd)
    angle = angle/np.sqrt(np.sum(angle**2))  # micron unit

    # let L = {x: x=loc+angle*a for some a}
    # find intersection between L and box
    w1,w2 = box_line_intersect(np.zeros(nd),box,loc,angle)

    # get length
    L = np.linalg.norm(w1-w2)  # micron length.
    numrols = npr.poisson(L*rolplength)

    # get positions
    alphas=npr.rand(numrols)
    rolpos = alphas[:,None]*w1[None,:]+(1-alphas[:,None])*w2[None,:]

    return loc,angle,rolpos
