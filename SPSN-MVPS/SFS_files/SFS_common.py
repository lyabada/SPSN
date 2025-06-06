
import torch


def World2Camera_tens(pose_c2w,pts):
    # pose_c2w[:3,1:3] *= -1
    
    world_to_cam = torch.linalg.inv(pose_c2w)
    pts_camera_space = world_to_cam @ pts    
    return pts_camera_space


def calculate_projection_matrices_tens(K, extrinsic):
    projections = []
    for E in extrinsic:
        P = K @ E[:3, :4]
        projections.append(P)
    return projections


def get_occupations_pts_sfs_tens(pts_tens,projections,masks,extrinsic,K, nb_views):
        #pts = pts_tens.to('cpu').numpy()
        filled = []
        i=0
        for P, im in zip(projections, masks):
            point_dans_camera_tens=World2Camera_tens(extrinsic[i],pts_tens)
            i+=1
            uvs = K @ point_dans_camera_tens[:3, :] #conversion du 3d vers 2d
            
            uvs = uvs/uvs[2, :].clone()
            uvs = torch.round(uvs).int()
            #print(uvs[0][0],uvs[1][0])
            imgH,imgW= masks.shape[1:]
            x_good = torch.logical_and(uvs[0, :] >= 0, uvs[0, :] < imgW)
            y_good = torch.logical_and(uvs[1, :] >= 0, uvs[1, :] < imgH)
            good = torch.logical_and(x_good, y_good)
            indices = torch.where(good)[0]
            fill = torch.zeros_like(pts_tens[0])
            sub_uvs = uvs[:2, indices].long()
            res = im[sub_uvs[1, :], sub_uvs[0, :]]
            fill[indices] = res 
            filled.append(fill)
        filled = torch.vstack(filled)

        occupancy_nb = torch.sum(filled, dim=0)
        occupancy = (occupancy_nb >= nb_views).float()

        #pot2 min 0.7
        #bear min -1
        #occupancy[pts_tens[2,:]<-0.7]=0.
        return occupancy, occupancy_nb 
    

def get_pixels_pts_in_image_tens(pts_tens, extrinsic, K,imgH,imgW):
        
        point_dans_camera=World2Camera_tens(extrinsic,pts_tens)
        uvs = K @ point_dans_camera[:3, :]
        uvs = uvs/uvs[2, :].clone()
        uvs = torch.round(uvs).long()
    
        y_good = torch.logical_and(uvs[0, :] >= 0, uvs[0, :] < imgW)
        x_good = torch.logical_and(uvs[1, :] >= 0, uvs[1, :] < imgH)
        good = torch.logical_and(x_good, y_good)
        return uvs, good
        #return uvs, good


import numpy as np
def save_new_points_in_file(s,nb_pts_epoch,path_file):
        x, y, z = np.mgrid[-s:s:complex(0,s*2), -s:s:complex(0,s*2), -s:s:complex(0,s*2)]
        # x, y, z = np.mgrid[:s, :s, :s]
        pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).astype(np.float32)
        pts = pts.T
        nb_points_init = pts.shape[0]
        xmax, ymax, zmax = np.max(pts, axis=0)
        pts[:, 0] /= xmax
        pts[:, 1] /= ymax
        pts[:, 2] /= zmax
        center = pts.mean(axis=0)
        #pts -= center
        #pts *= pts_fact

        np.random.shuffle(pts)
        pts = np.vstack((pts.T, np.ones((1, pts.shape[0]),dtype=np.float32)))
        

        with open(path_file, 'wb') as f:
            for i_epoch in range(0, int(pts.shape[1]), nb_pts_epoch):
                np.save(f, np.array(pts[:,i_epoch:i_epoch+nb_pts_epoch]))
