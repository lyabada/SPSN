''' From PSNERF Code : https://github.com/ywq/psnerf'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import (
    get_mask, origin_to_world,image_points_to_ray)

epsilon = 1e-6


def get_sphere_intersection(cam_loc, ray_directions, r = 1.0):
    # Input: n_images x 4 x 4 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays

    n_imgs, n_pix, _ = ray_directions.shape
    cam_loc = cam_loc.unsqueeze(-1)
    ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze()
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2,1) ** 2 - r ** 2)

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0
    
    sphere_intersections = torch.zeros(n_imgs * n_pix, 2).to(cam_loc.device).float()
    sphere_intersections[mask_intersect] = torch.sqrt(under_sqrt[mask_intersect]).unsqueeze(-1) * torch.Tensor([-1, 1]).to(cam_loc.device).float()
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[mask_intersect].unsqueeze(-1)

    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_min(0.0)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)

    return sphere_intersections, mask_intersect


def unisurf(model,device, pixels, camera_mat, world_mat, 
                scale_mat, it=100000):
        # Get configs
        batch_size, n_points, _ = pixels.shape
        rad = 2.0
        ray_steps = 64
        
        #cow
        depth_range = torch.tensor([34,36])
             
        
        # Prepare camera projection
        camera_world = origin_to_world(
            n_points, camera_mat, world_mat, scale_mat
        )
        ray_vector = image_points_to_ray(pixels, camera_mat, world_mat)
        ray_vector = ray_vector/ray_vector.norm(2,2).unsqueeze(-1)
        
        
        
        # Find surface
        with torch.no_grad():
            d_i = ray_marching(
                camera_world, ray_vector, model,
                n_secant_steps=8, 
                n_steps=[int(ray_steps),int(ray_steps)+1], 
                rad=rad,
                depth_range=depth_range,
            )

        # Get mask for where first evaluation point is occupied
        mask_zero_occupied = d_i == 0
        d_i = d_i.detach()

        # Get mask for predicted depth
        mask_pred = get_mask(d_i).detach()
        
        with torch.no_grad():
            dists =  torch.ones_like(d_i).to(device)
            dists[mask_pred] = d_i[mask_pred]
            dists[mask_zero_occupied] = 0.
            network_object_mask = mask_pred & ~mask_zero_occupied
            network_object_mask = network_object_mask[0]
            dists = dists[0]

        # Project depth to 3d poinsts
        camera_world = camera_world.reshape(-1, 3)
        ray_vector = ray_vector.reshape(-1, 3)
        
        points = camera_world + ray_vector * dists.unsqueeze(-1)
        points = points.view(-1,3)

        ## normal, diff_norm
        surface_mask = network_object_mask.view(-1)
        surface_points = points[surface_mask]
        N = surface_points.shape[0]
        
        surface_points_neig = surface_points + (torch.rand_like(surface_points) - 0.5) * 0.01      
        pp = torch.cat([surface_points, surface_points_neig], dim=0)
    
        g = model.gradient(pp) 
        normals_ = g[:, 0, :] / (g[:, 0, :].norm(2, dim=1).unsqueeze(-1) + 10**(-5))
        norm_pred = torch.zeros_like(points)
        norm_pred[surface_mask] = normals_[:N]

        
       
        
        out_dict = {
            'mask_pred': network_object_mask,
            'normal_pred': norm_pred.reshape(batch_size, -1, 3),
            'it':it
        }
        
        return out_dict


def ray_marching(ray0, ray_direction, model, c=None,
                             tau=0.5, n_steps=[128, 129], n_secant_steps=8,
                             depth_range=[25.,40.], max_points=3500000, rad=1.0,
                             clip=False):
        ''' Performs ray marching to detect surface points.

        The function returns the surface points as well as d_i of the formula
            ray(d_i) = ray0 + d_i * ray_direction
        which hit the surface points. In addition, masks are returned for
        illegal values.

        Args:
            ray0 (tensor): ray start points of dimension B x N x 3
            ray_direction (tensor):ray direction vectors of dim B x N x 3
            model (nn.Module): model model to evaluate point occupancies
            c (tensor): latent conditioned code
            tay (float): threshold value
            n_steps (tuple): interval from which the number of evaluation
                steps if sampled
            n_secant_steps (int): number of secant refinement steps
            depth_range (tuple): range of possible depth values (not relevant when
                using cube intersection)
            method (string): refinement method (default: secant)
            check_cube_intersection (bool): whether to intersect rays with
                unit cube for evaluation
            max_points (int): max number of points loaded to GPU memory
        '''
        # Shotscuts
        batch_size, n_pts, D = ray0.shape
        device = ray0.device
        tau = 0.5
        n_steps = torch.randint(n_steps[0], n_steps[1], (1,)).item()

            
        depth_intersect, _ = get_sphere_intersection(ray0[:,0], ray_direction, r=rad)
        d_intersect = depth_intersect[...,1]            
        
        d_proposal = torch.linspace(
            0, 1, steps=n_steps).view(
                1, 1, n_steps, 1).to(device)
        d_proposal = depth_range[0] * (1. - d_proposal) + d_intersect.view(1, -1, 1,1)* d_proposal

        p_proposal = ray0.unsqueeze(2).repeat(1, 1, n_steps, 1) + \
            ray_direction.unsqueeze(2).repeat(1, 1, n_steps, 1) * d_proposal

        # Evaluate all proposal points in parallel
        with torch.no_grad():
            val = torch.cat([(
                model(p_split, only_occupancy=True) - tau)
                for p_split in torch.split(
                    p_proposal.reshape(batch_size, -1, 3),
                    int(max_points / batch_size), dim=1)], dim=1).view(
                        batch_size, -1, n_steps)

        if clip:
            val[(p_proposal>1).any(-1).view(batch_size, -1, n_steps)]=-1
            val[(p_proposal<-1).any(-1).view(batch_size, -1, n_steps)]=-1
        # Create mask for valid points where the first point is not occupied
        mask_0_not_occupied = val[:, :, 0] < 0

        # Calculate if sign change occurred and concat 1 (no sign change) in
        # last dimension                    [B,N,S]
        sign_matrix = torch.cat([torch.sign(val[:, :, :-1] * val[:, :, 1:]),
                                 torch.ones(batch_size, n_pts, 1).to(device)],
                                dim=-1)
        cost_matrix = sign_matrix * torch.arange(
            n_steps, 0, -1).float().to(device)     # -n for min cost

        # Get first sign change and mask for values where a.) a sign changed
        # occurred and b.) no a neg to pos sign change occurred (meaning from
        # inside surface to outside)
        values, indices = torch.min(cost_matrix, -1)
        mask_sign_change = values < 0
        mask_neg_to_pos = val[torch.arange(batch_size).unsqueeze(-1),
                              torch.arange(n_pts).unsqueeze(-0), indices] < 0

        # Define mask where a valid depth value is found
        mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied 

        # Get depth values and function values for the interval
        # to which we want to apply the Secant method
        # get X_n, X_n+1 and density of them
        n = batch_size * n_pts
        d_low = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_low = val.view(n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
            batch_size, n_pts)[mask]
        indices = torch.clamp(indices + 1, max=n_steps-1)
        d_high = d_proposal.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]
        f_high = val.view(
            n, n_steps, 1)[torch.arange(n), indices.view(n)].view(
                batch_size, n_pts)[mask]

        ray0_masked = ray0[mask]
        ray_direction_masked = ray_direction[mask]

        # write c in pointwise format
        if c is not None and c.shape[-1] != 0:
            c = c.unsqueeze(1).repeat(1, n_pts, 1)[mask]
        
        # Apply surface depth refinement step (e.g. Secant method)
        d_pred = secant(model,
            f_low, f_high, d_low, d_high, n_secant_steps, ray0_masked,
            ray_direction_masked, tau)

        # for sanity
        d_pred_out = torch.ones(batch_size, n_pts).to(device)
        d_pred_out[mask] = d_pred
        d_pred_out[mask == 0] = np.inf
        d_pred_out[mask_0_not_occupied == 0] = 0
        return d_pred_out

def secant(model, f_low, f_high, d_low, d_high, n_secant_steps,
                          ray0_masked, ray_direction_masked, tau, it=0):
        ''' Runs the secant method for interval [d_low, d_high].

        Args:
            d_low (tensor): start values for the interval
            d_high (tensor): end values for the interval
            n_secant_steps (int): number of steps
            ray0_masked (tensor): masked ray start points
            ray_direction_masked (tensor): masked ray direction vectors
            model (nn.Module): model model to evaluate point occupancies
            c (tensor): latent conditioned code c
            tau (float): threshold value in logits
        '''
        d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        for i in range(n_secant_steps):
            p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_direction_masked
            with torch.no_grad():
                f_mid = model(p_mid,  batchwise=False,
                                only_occupancy=True, it=it)[...,0] - tau
            ind_low = f_mid < 0
            ind_low = ind_low
            if ind_low.sum() > 0:
                d_low[ind_low] = d_pred[ind_low]
                f_low[ind_low] = f_mid[ind_low]
            if (ind_low == 0).sum() > 0:
                d_high[ind_low == 0] = d_pred[ind_low == 0]
                f_high[ind_low == 0] = f_mid[ind_low == 0]

            d_pred = - f_low * (d_high - d_low) / (f_high - f_low) + d_low
        return d_pred
    