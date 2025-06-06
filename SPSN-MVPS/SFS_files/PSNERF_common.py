import torch
import numpy as np
from SFS_files.SFS_common import get_occupations_pts_sfs_tens


#----------------------------------------------
#--------------SPSN-MVPS------------------------
#----------------------------------------------

def get_occupations_pts_model_tens(pts, model):
        model.eval()
        with torch.no_grad():
            p = pts[:3,:].T
            val = model(p, only_occupancy=True)
            val = val[:,0]
            #val[val>0.5] = 1
            #val[val<=0.5] = 0
            #val = val.to('cpu').numpy()
            return val
    

def get_points_in_surface(pts, rays,occ_in, dist,model=None, use_model=False):
        ps_delta = pts + rays.T  * dist
        ps_delta = torch.vstack((ps_delta, torch.ones_like(ps_delta[0])))
        occ_delta = get_occupations_pts_sfs_tens(ps_delta)[0] if not use_model or model==None else get_occupations_pts_model_tens(ps_delta,model)
        
        #pos_point = (occ_in * (1-occ_delta))  if dist < 0 else ((1-occ_in) * occ_delta)
        pos_point = ((occ_in -0.5) * (occ_delta-0.5))+0.5
        return ps_delta, pos_point, occ_delta

def get_points_decot_surface(pts_in, pts_out, model, nb_dicot=8):

        pts_in = pts_in.clone()
        pts_out = pts_out.clone()
        occ_mid =  []
        for iter in range(nb_dicot):
            pts_mid = (pts_in+pts_out)/2
            
            pp = torch.vstack((pts_mid, torch.ones_like(pts_mid[0])))
            
            occ_mid = get_occupations_pts_model_tens(pp,model) -0.5
            
            ind_low = occ_mid < 0
            if ind_low.sum() > 0:
                pts_out[:,ind_low] = pts_mid[:,ind_low]
            
            if (ind_low == 0).sum() > 0:
                pts_in[:,ind_low == 0] = pts_mid[:,ind_low == 0]
                
        #print("fdeco : ",occ_mid )
        return pts_in
    
def get_points_Regula_Falsi_surface(pts_in, pts_out, model, nb_dicot=8):

        pts_in = pts_in.clone()
        pts_out = pts_out.clone()

        f_out = get_occupations_pts_model_tens(pts_out,model) -0.5
        f_in = get_occupations_pts_model_tens(pts_in,model) -0.5
        
        for iter in range(nb_dicot):
            #pts_mid = (pts_in+pts_out)/2
            
            # out <=> low
            # in <=> high
            
            pts_mid = - f_out * (pts_in - pts_out) / (f_in - f_out) + pts_out
            

            pp = torch.vstack((pts_mid, torch.ones_like(pts_mid[0])))
            
            f_mid = get_occupations_pts_model_tens(pp,model) -0.5
            
            ind_low = f_mid < 0
            if ind_low.sum() > 0:
                pts_out[:,ind_low] = pts_mid[:,ind_low]
                f_out[ind_low] = f_mid[ind_low]
            
            if (ind_low == 0).sum() > 0:
                pts_in[:,ind_low == 0] = pts_mid[:,ind_low == 0]
                f_in[ind_low == 0] = f_mid[ind_low == 0]
        #print("f mid: ",f_mid)
        return pts_in


#---------------------------------------------
#--------------PSNERF-Common------------------
#---------------------------------------------



def to_pytorch(tensor, return_type=False):
    ''' Converts input tensor to pytorch.

    Args:
        tensor (tensor): Numpy or Pytorch tensor
        return_type (bool): whether to return input type
    '''
    is_numpy = False
    if type(tensor) == np.ndarray:
        tensor = torch.from_numpy(tensor)
        is_numpy = True

    tensor = tensor.clone()
    if return_type:
        return tensor, is_numpy
    return tensor

def image_points_to_ray(image_points, camera_mat, world_mat):
    # image_points (tensor):  B x N x 2
    image_points = to_pytorch(image_points)
    batch_size, n_pts, dim = image_points.shape
    assert(dim == 2)
    device = image_points.device
    depth = torch.ones(batch_size, n_pts, 1).to(device)
    pixels, is_numpy = to_pytorch(image_points, True)
    depth = to_pytorch(depth)
    camera_mat = to_pytorch(camera_mat)
    world_mat = to_pytorch(world_mat)
    

    p_trans = (pixels-camera_mat[0,:2,2])/camera_mat[0,0,0]
    p_trans = torch.cat([p_trans, torch.ones_like(p_trans[...,:1])], dim=2) 
    p_trans[:,:,:3] *= depth
    ray_dirs = torch.einsum('bij,bnj->bni',world_mat[:,:3,:3], p_trans)

    
    if is_numpy:
        ray_dirs = ray_dirs.numpy()
    return ray_dirs


def get_ray_from_pixels(pixels, pose_c2w, K ):
        points_pixel = pixels.T
        
        ray_vector = image_points_to_ray(points_pixel.unsqueeze(0), K.unsqueeze(0), pose_c2w.unsqueeze(0)).float()
        ray_vector = ray_vector/ray_vector.norm(2,2).unsqueeze(-1)
        #ray_vector = ray_vector.numpy()[0]
        return ray_vector



def shuffle_occupancy_points(occupations,occ_val,unocc_val,occ_pers,unocc_pers):
        occupied_indices = torch.where(occupations == occ_val)[0]
        unoccupied_indices = torch.where(occupations == unocc_val)[0]
        sample_size = int(len(unoccupied_indices) * unocc_pers)
        sample_size2 = int(len(occupied_indices) * occ_pers)
        indices = torch.randperm(len(unoccupied_indices))[:sample_size]
        sampled_unoccupied_indices = unoccupied_indices[indices]
        indices = torch.randperm(len(occupied_indices))[:sample_size2]
        sampled_occupied_indices = occupied_indices[indices]
        final_indices = torch.cat((sampled_occupied_indices, sampled_unoccupied_indices))
        indices = torch.randperm(len(final_indices))
        final_indices = final_indices[indices]

        
        return final_indices
        