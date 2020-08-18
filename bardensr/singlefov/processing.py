import numpy as np
import scipy as sp
import scipy.spatial.distance
from .. import kernels
import itertools
import numpy.random as npr
import skimage

import collections


def downsample_nd(X,ns):
    for (i,n) in enumerate(ns):
        X=downsample(X,n,axis=i)
    return X

def downsample(X,n,axis=0):
    shp=list(X.shape)

    # get divisible by n
    slices=[slice(0,None) for i in range(len(shp))]
    slices[axis]=slice(0,n*(shp[axis]//n))
    X=X[tuple(slices)]

    # reshape
    newshp=list(X.shape)
    newshp.insert(axis+1,n)
    newshp[axis]=newshp[axis]//n
    X=np.reshape(X,newshp)

    # average
    X=np.mean(X,axis=axis+1)

    # done!
    return X





def rm_edge_spots(target_j, orig_shape, m = 0):  
    '''
    function remove edge spots. 
    target_j is (S, m1, m2, m3), 
    where S is the total number of blobs detected
    returns (S', m1, m2, m3) where S' <= S
    '''
    rm_id = []
    for s in range(target_j.shape[0]):  # flag indicates spots pass 
        flag_1 = ((target_j[s][0] >= 0 + m) and (target_j[s][0] < orig_shape[0] - m))
        flag_2 = ((target_j[s][1] >= 0 + m) and (target_j[s][1] < orig_shape[1] - m))
        flag_3 = ((target_j[s][2] >= 0) and (target_j[s][2] < orig_shape[2]))
        if (flag_1 and flag_2 and flag_3) == False:
            rm_id.append(s)            
    out = np.delete(target_j, rm_id, axis = 0)
    return(out)


def coefficient_of_determination(x, y):   # input 3d array 
    x = x.ravel()
    y = y.ravel()
    x_norm = (x-x.min()) / (x-x.min()).max()
    y_norm = (y-y.min()) / (y-y.min()).max()    
    out = np.corrcoef(x_norm, y_norm)[0, 1]**2
    return(out)


def svd(loc_model, alpha,  Y_j, m):
    '''
    run svd for all the spots detected for this barcode. 
    loc_model is (S, 3)
    return the dictionary with svd results 
    including V, U, r2 for every spot
    '''
    R, C = Y_j.shape[-2:]
    N = R*C    
    U_list = []
    V_list = []
    r2_list = []
    for s in range(len(loc_model)):
        single_img = Y_j[(loc_model[s, 0]-m):(loc_model[s, 0]+m), 
                         (loc_model[s, 1]-m):(loc_model[s, 1]+m), 
#                          (loc_model[s, 2]-m):(loc_model[s, 2]+m), 
                         :
                        ]    # (2m, 2m, 2m, R, C)
        m1, m2, m3 = single_img.shape[:3]
        single_img = single_img.reshape(m1*m2*m3, N).T   # (N, m1*m2*m3)        
        
        single_img_svd = (single_img/alpha.reshape(N)[:, None])   # (N, m1*m2*m3)          
        U, sigma, Vh = sp.linalg.svd(single_img_svd)  # U:(N,N), Vh:(m1*m2*m3, m1*m2*m3)
        negative = int(Vh[0, :].reshape(m1,m2,m3)[int(m1/2),int(m2/2),int(m3/2)] < 0)
        
        U_list.append((-1)**negative*U[:, 0].reshape(R, C))  
        V_list.append((-1)**negative*Vh[0, :].reshape(m1,m2,m3))                   
        r2 = coefficient_of_determination(single_img,   # (N, m1*m2*m3)
                                          U_list[-1].ravel()[:, None]*V_list[-1].ravel()[None, :]
                                         )
        r2_list.append(r2)
    return dict(
        temporal = U_list,
        spatial = V_list,
        r2 = r2_list,
    )
    

def cleaned_img_svd(Xsub, model, thre, j, tile, m=5):  # for one/ patch. 
    '''
    output: 
        svd_tile_results - dictiorary of svd results, could be empty. 
    input: 
        Xsub - tile (look up)
        model - model_fine
        thre - tile threshold
        j - this barcode. 
    
    '''
    # setup
    M1, M2, M3, R, C = Xsub.shape
    M = M1*M2*M3    
    N = R*C
    
    # cleaned image
    F_tilde = np.delete(np.array(model.F_blurred).reshape(M, -1), j, axis = -1)   # Mx(J-1)
    G_tilde = np.delete(np.array(model.frame_loadings()).reshape(N, -1), j, axis = -1)  # Nx(J-1)    
    Y_tilde = F_tilde @ G_tilde.T + np.array(model.a).reshape(M)[:, None]+ np.array(model.b).reshape(N)[None, :]
    Y_tilde = Y_tilde.reshape(M1, M2, M3, R, C)
    Y_j = Xsub - Y_tilde    # this is the same size as lookup tile.   
    
    # detect blobs, get svd for each blob. 
    Fs = model.F_scaled(blurred = False)[tile.grab]   # same size as look tile
    Fs_blurred = model.F_scaled(blurred = True)[tile.grab] 
    loc_model = skimage.feature.peak_local_max(Fs[:,:,:,j],  # finding the blobss in the look tile...
                                               threshold_abs = thre,
                                               min_distance = 3, 
                                               exclude_border=False)
    loc_model = rm_edge_spots(target_j = loc_model.astype(int), 
                              orig_shape = Fs.shape,
                              m = m
                              ) 
    
    if len(loc_model) >0:        
        svd_results = svd(loc_model = loc_model, alpha = np.array(model.alpha), Y_j = Y_j[tile.grab], m = m)                
#         single_imgs_list = [Fs_blurred[x[0]-m:x[0]+m, x[1]-m:x[1]+m, :, j] for x in loc_model]  # F original image (j, tile)     

        single_imgs_list = []
        for x in loc_model:
            single_imgs_list.append(Fs_blurred[x[0]-m:x[0]+m, x[1]-m:x[1]+m, :, j])
            if single_imgs_list[-1].shape[0] != 2*m:
                print(x, x[0]-m, x[0]+m, x[1]-m, x[1]+m)
            if single_imgs_list[-1].shape[1] != 2*m:
                print(x, x[0]-m, x[0]+m, x[1]-m, x[1]+m)            
        svd_results['imgs'] = single_imgs_list  # add the original Fs into the dict, each is (m1,m2,m3)        
        coord = []  # coords on the original fov scale
        for x in loc_model:
            coord_x = x[0] + tile.put[0].start
            coord_y = x[1] + tile.put[1].start
            coord_z = x[2] + tile.put[2].start            
            coord.append(np.array((coord_x, coord_y, coord_z)))        
        svd_results['coord'] = coord  # add the coordinates of this image. 
    else:
        svd_results = dict()
    return(svd_results)





    
    
    
    