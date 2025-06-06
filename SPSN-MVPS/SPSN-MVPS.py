# notepad replace lines : ^[ba|st|en|>>].*?\r\n
if __name__ == '__main__':
    import numpy as np
        
    import imageio.v2 as imageio
    import os
    import json

    from SFS_files.SFS_common import *
    from SFS_files.PSNERF_common import *
    from model.training_sfs import render_visdata
    #----------------------------------------------------------------------------
    from re import I
    import numpy as np
    import torch
    import torch.nn as nn
    import os, sys, time
    import logging
    import scipy.io
    
    import torch.optim as optim
    import model as mdl
    from torch.types import Device
    from utils.tools import set_debugger
    from matplotlib import pyplot as plt
    import copy


    l1_loss = nn.L1Loss(reduction='sum')
    l1_loss_None_reduce = nn.L1Loss(reduction='none', reduce=None)
    l2_loss = nn.MSELoss(reduction='sum')
    

    object_name = 'cow'
    model_file_name = 'model_spsn'
    basedir = '../dataset/'+object_name
    para = json.load(open(os.path.join(basedir,'params.json')))
    out_dir = '../out/'+object_name+'/test1/'
    visualize_path = os.path.join(out_dir, 'images')
    
    sfs_points_file = '../points_grid.npy'
    nb_pts_epoch = 120000
    n_max_network_queries = 120000
    delta_d = 0.6
    delta_d_added = delta_d 
    save_new_points = False
    s = 150
    pts_fact = 2.0
    use_model = True
    lr = 0.0001
    weight_decay = 0.0
    eps = 0.0001
    
    #Angle min between Z-axis and used normals (0:any vector, 90 all vectors)
    min_norm_angle=torch.cos(torch.deg2rad(torch.tensor(80)))  
    
    logger_py = logging.getLogger(__name__)
    np.random.seed(42)
    torch.manual_seed(42)
    set_debugger()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Thanks to PS-NeRF and unisurf for this
    model = mdl.NeuralNetwork().to(device) 
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[4000, 8000], 
        gamma=0.5, 
        last_epoch=-1)
    #test_loader = dl.get_dataloader(cfg, mode='test')
    epoch=0
    
    
    
    view_test = para['view_test']
    view_train = para['view_train']
    use_all_view = False


    poses = para['pose_c2w']
    K_tens = torch.tensor(para['K']).float().to(device)
    extrinsic_tens = [torch.tensor(pose).float().to(device) for pose in poses]
    for m in range(len(extrinsic_tens)):
        x=extrinsic_tens[m]
        x[:3,1:3] *= -1
        extrinsic_tens[m]=x
    extrinsic_tens = torch.stack(extrinsic_tens)

    checkpoint_path = os.path.join(out_dir,'models')
    checkpoint_file = model_file_name+'.pt'

    checkpoint_io = mdl.CheckpointIO(checkpoint_path, model=model, optimizer=optimizer)
    
    try:
        load_dict = checkpoint_io.load(checkpoint_file)
        print("Chekpoint loaded :",checkpoint_path,"/",checkpoint_file)
    except FileExistsError:
        load_dict = dict()
        load_sfs=True
        print("checkpoint :",checkpoint_file," not found in : ",checkpoint_path)
    
    epoch_it = load_dict.get('epoch_it', -1)    
    it = load_dict.get('it', -1)
    time_all = load_dict.get('time_all', 0)

    os.makedirs(visualize_path, exist_ok=True)
    
    imgs,masks,norm_masks, normals, normals_gt=[],[],[],[],[]

    nb_images = nb_images = len(view_train)+len(view_test)
    for i in range(nb_images):
        img = imageio.imread(os.path.join(basedir,'img/view_{:02d}.png'.format(i+1)))#, cv2.IMREAD_COLOR)
        norm_gt = np.load(os.path.join(basedir,'normal_gt/outnpy/view_{:02d}.npy'.format(i+1)))

        # Crop normal maps from Diligent dimensions (512x612) to PS-NeRF dimensions (400x400)
        # The normal (512x612) maps is availabale in : https://sites.google.com/site/photometricstereodata/mv
        # This step can be improved by using more accurate normal maps from PS.
        # To change the object it is necessary to change the position of the crop
        #cow
        bor1 = 36
        bor2 = 106
        norm = mat = scipy.io.loadmat(os.path.join(basedir,'normal_ps_TIP19Li/matlab/'+object_name+'PNG_Normal_TIP19Li_view{:01d}.mat'.format(i+1)))
        norm = norm['Normal_est']
        norm = norm[bor1:bor1+400,bor2:bor2+400]
        #----------------------------------------------------------


        mask = np.array(imageio.imread(os.path.join(basedir,'mask/view_{:02d}.png'.format(i+1)), pilmode="RGB"))
        norm_mask = np.array(imageio.imread(os.path.join(basedir,'norm_mask/view_{:02d}.png'.format(i+1)), pilmode="RGB"))
        imgs.append(img)
        normals.append(norm)
        normals_gt.append(norm_gt)
        masks.append(mask)
        norm_masks.append(norm_mask)

        
    imgs = (np.array(imgs) / 255.).astype(np.float32)
    masks = (np.array(masks) / 255.).astype(np.float32)  

    norm_masks = (np.array(norm_masks) / 255.).astype(np.float32)  
    masks = masks[:,:,:,0]
    norm_masks = norm_masks[:,:,:,0]

    imgs = imgs*masks[...,None] #+ (1.-masks[...,None])
    normals = np.array(normals).astype(np.float32).transpose(0,3,1,2)
    normals_gt = np.array(normals_gt).astype(np.float32).transpose(0,3,1,2)
    
    imgs = torch.tensor(imgs,dtype=torch.float).to(device)
    masks = torch.tensor(masks,dtype=torch.float).to(device)
    norm_masks = torch.tensor(norm_masks,dtype=torch.float).to(device)
    normals = torch.tensor(normals,dtype=torch.float).to(device)
    normals_gt = torch.tensor(normals_gt,dtype=torch.float).to(device)
    
    imgH,imgW= imgs.shape[1:3]

    imgs_test = imgs[view_test]
    masks_test = masks[view_test]
    norm_masks_test = norm_masks[view_test]
    extrinsic_test = extrinsic_tens[view_test]
    normals_test = normals[view_test]
    normals_gt_test = normals_gt[view_test]


    if(not use_all_view):
        imgs = imgs[view_train]
        masks = masks[view_train]
        norm_masks = norm_masks[view_train]
        normals = normals[view_train]
        normals_gt = normals_gt[view_train]
        extrinsic_tens = extrinsic_tens[view_train]
        nb_images = imgs.shape[0]

    
    projections_tens=calculate_projection_matrices_tens(K_tens,extrinsic_tens)
    #-----------------------------------------------------------------------------------------

    def loss_fonction(weight_pos, alpha_true,alpha_pred, diff_norm, norm_pred, normal_ps, it):
        #If the model converges to occ_mask = 0 and occ_in = 1, it is necessary to either extend the SFS phase 
        # or increase the weight_loss_occ_in_obj during the gradient step.
        weight_norm = 0.4 if it <= 1000 else 3.0 #if it <= 750 else 5.0
        weight_grad = 0.01 if it <= 2000 else 0.0002

        
        #exponential function f(x) = 0.002**x+0.005  // x: 0--> 1 y: 1-->0.005
        #iterations: 0.0 --> 50000 ==> x=iter/20000 weight(y): 1.0-->~0.005
        x = it/50000
        weight_loss_occ_in_obj = 0.002**x+0.05
        
        weight_loss_occ_mask_0 = 1.0 #if it <= 2000 else 2.0
        
        alpha_pred_in_obj = alpha_pred[alpha_true>=0.5]
        #weight_pos_in_obj = weight_pos[alpha_true[:,0]>=0.5]
        alpha_true_in_obj = alpha_true[alpha_true>=0.5]
        
        loss = torch.tensor(0.0).to(device).float()

        if alpha_true[alpha_true<=0.5].shape[0]==0:
                loss_occ_mask_0 = torch.tensor(0.0).to(device).float()
        else :
                loss_occ_mask_0 = (l1_loss(alpha_pred[alpha_true<=0.5],alpha_true[alpha_true<=0.5]))/float(alpha_true[alpha_true<=0.5].shape[0])
                loss += weight_loss_occ_mask_0 * loss_occ_mask_0
        
        if alpha_true_in_obj.shape[0]==0:
                loss_occ_in_obj = torch.tensor(0.0).to(device).float()
        else :
                #nb_points_out(mask) > nb_points_in;  about 10 times larger, but only points close to the object are important.
                loss_occ_in_obj = ((l2_loss(alpha_pred_in_obj,alpha_true_in_obj))/float(alpha_true_in_obj.shape[0]))
                loss += weight_loss_occ_in_obj * loss_occ_in_obj 
                
        if diff_norm is None or diff_norm.shape[0]==0:
                grad_loss = torch.tensor(0.0).to(device).float()
        else :
                grad_loss = diff_norm.mean()
                loss +=  weight_grad * grad_loss 
        
        if norm_pred is None or normal_ps.shape[1]==0 :
                normal_loss = torch.tensor(0.0).to(device).float()
        else :
                #normal_loss = l2_loss(norm_pred[0], normal_ps[0]) / float(normal_ps.shape[1])
                #loss += weight_norm * normal_loss       
                  
                
                dot_product = torch.einsum ('xij, xij -> ij', norm_pred, normal_ps)
                dot_product = dot_product.sum(-1)    
                dot_product = dot_product.clip(-1+eps, 1-eps)
                normal_loss = torch.arccos(dot_product) #* 180 / np.pi
                normal_loss = normal_loss.sum() / float(norm_pred.shape[1])
                loss += weight_norm * normal_loss 
                
                #-------------------to show normal MAE in degree------------------
                normal_loss  *=  (180 / torch.pi)
                
        return loss, loss_occ_in_obj, loss_occ_mask_0, grad_loss, normal_loss

    #------------------------------------------------------------------------
    # Save the grid points file only the first time if the file doesn't exist and save_new_points is False
    # , or every time if save_new_points is True.
    if(save_new_points or not os.path.exists(sfs_points_file)) :
        print("Start, Save the grid points file")
        save_new_points_in_file(s,nb_pts_epoch,sfs_points_file)
        print("End, Save the grid points file")
    #------------------------------------------------------------------------
        
    loss, loss_occ_mask_0, loss_occ_in_obj,grad_loss,normal_loss, nb = 0,0,0,0,0,0
    time_iters = 0.
    best_error = 99999.
    best_model = model
    
    while(True):   #If the end of the grid file is reached, reload the file.
        if(save_new_points) :
            print("Start, Save the grid points file")
            save_new_points_in_file(s,nb_pts_epoch,sfs_points_file)
            print("End, Save the grid points file")
        try:
            with open('../points_grid.npy', 'rb') as f:
                while(True) :
                    time_start = time.time()
                    epoch_it += 1
                    #if it > 1000 : delta_d = 0.15

                    #load points from files-----use save_new_points=True to create new file---------
                    sub_pts = np.load(f) 
                    sub_pts=torch.tensor(sub_pts,dtype=torch.float).to(device)
                    
                    #----------change the points interval [-1,1] to [-pts_fact,pts_fact]
                    sub_pts[:3,:] *= pts_fact
                    
                    #----------add noise to points each loading of the file------------
                    #----The noise is shifted by half the distance between neighboring points----
                    rand_shift = (((torch.rand(3,sub_pts.shape[1])-0.5))*pts_fact/(1.*s)).to(device)
                    sub_pts[:3,:] = sub_pts[:3,:] + (rand_shift*2)

                    #--------Compute SFS Occupancy---------------
                    occupations, occupations_nb =  get_occupations_pts_sfs_tens(sub_pts,projections_tens,masks,extrinsic_tens,K_tens,masks.shape[0])

                    #-------Use points that have at least one projection on an object mask in any view (i.e., occupancy > 0),------
                    # ---------where occupancy represents the number of intersections with masks across all views.------------
                    sub_pts = sub_pts[:,occupations_nb>=0]
                    occupations = occupations[occupations_nb>=0]
                    
                    #-------shuffle occupied and not occupied points---------
                    final_indices = shuffle_occupancy_points(occupations,1,0,1,0.1)
                    sub_pts = sub_pts[:,final_indices]
                    
                    #------------------occupations=occupations[final_indices] -----------------------
                    # ----apply a forward on the model to get the predected occupancy of points------
                    occupations =  get_occupations_pts_model_tens(sub_pts,model)

                    #-------for image in train views---------
                    for i in range(nb_images):  
                        #project 3D point from grid to pixels (u,v) position in image (word2image)
                        uvs, good = get_pixels_pts_in_image_tens(sub_pts, extrinsic_tens[i], K_tens,imgH,imgW)
                        
                        #---if there is at least one point projected on the image (u,v in 0..400,0..400)---- 
                        if any(good):
                            indices = torch.where(good)[0]
                            occ_tens=occupations[indices]
                            ps = sub_pts[:3,indices]
                            pixels = uvs[:2, indices]
                            
                            #thaks to PSNERF code :)
                            ray_vector = get_ray_from_pixels(pixels,extrinsic_tens[i],K_tens).squeeze().to(device)

                            
                            msk = masks[i]
                            msk_gt = msk[pixels[1],pixels[0]]

                            #------------weights occ--------------------------   
                            # -------- this function: get_points_in_surface is very important, -----
                            # -----it do the shifting and determine the points near to the surface -----
                            # ----- pos_point1 returns a value < 0.5 when the initial point is occupied but ---
                            # -- the shifted point is not — indicating that the point is close to the surface -----
                            ps_delta1, pos_point1, occ_delta1 = get_points_in_surface(ps,ray_vector,occ_tens, -delta_d, model=model, use_model=use_model)
                            
                            ps_delta1 = ps_delta1[:3,:]
                            weights_occ = pos_point1
                            p_in = ps[:,pos_point1<0.5]
                            p_out = ps_delta1[:,pos_point1<0.5]
                                
                            occ_sfs = get_occupations_pts_sfs_tens(torch.vstack((ps, torch.ones((1, ps.shape[1])).to(device))),projections_tens,masks,extrinsic_tens,K_tens,masks.shape[0])[0]
                            #weights_occ = weights_occ * occ_sfs
                            #-------------------------------------------------
                            ps = ps.T
                            ps_delta1 = ps_delta1.T
                            weights_occ = weights_occ.unsqueeze(1)
                            occ_sfs=occ_sfs.unsqueeze(1)
                            time_iters += time.time() - time_start
                            #We divide the points into parts with a maximum of n_max_network_queries per forward pass
                            for i2 in range(0, int(ps.shape[0]), n_max_network_queries):
                                    time_start = time.time()
                                    ray_vec_inv = -1*ray_vector[i2:i2+n_max_network_queries]
                                    weight_pos = (weights_occ[i2:i2+n_max_network_queries]<0.5)[:,0]
                                    msk_gt_it = (msk_gt[i2:i2+n_max_network_queries]>0.5)[0]
                                    
                                    # --- in the first time, the SFS object is out to the mask, ---
                                    # --- we can put the next line for rafinement. we can also add ---
                                    # --- the next line with big value of delta_d.---
                                    #if it > 10000:

                                    #weight_pos = weight_pos * msk_gt_it 

                                    #--------------------Normal points surface--------------------------------------------
                                    # ----- weight_pos is a mask of the points near to the surface------------
                                    surface_points = ps[i2:i2+n_max_network_queries][weight_pos].clone()

                                    #if surface_points.shape[0] < 5 : 
                                    #      continue

                                    surface_delta_points = ps_delta1[i2:i2+n_max_network_queries][weight_pos]
                                    
                                    surface_points = get_points_decot_surface(surface_points.T,surface_delta_points.T, model, nb_dicot=30).T

                                    # --------- The Regula Falsi method is recommended by the reviewers, thanks to them!---------
                                    #surface_points = get_points_Regula_Falsi_surface(surface_points.T,surface_delta_points.T, model, nb_dicot=30).T
                                    

                                    
                                    #occ_sfs_part = occ_sfs[i2:i2+n_max_network_queries].clone()
                                    occ_sfs_part = get_occupations_pts_sfs_tens(torch.vstack((surface_points.T, torch.ones((1, surface_points.shape[0])).to(device))),projections_tens,masks,extrinsic_tens,K_tens,masks.shape[0])[0]
                                    occ_sfs_part=occ_sfs_part.unsqueeze(1)
                                    
                                    #--------------------------------------------------------------------------------------
                                    model.train()
                                    optimizer.zero_grad()
                                    alpha = model(
                                         ps[i2:i2+n_max_network_queries], 
                                         only_occupancy=True)
                                    
                                    
                                    #--------------------------------------------------------
                                    N = surface_points.shape[0]
                                    #if(it > 1000 and N<20) : 
                                    #      continue
                                    
                                    it+=1
                                    surface_points_neig = surface_points + (torch.rand_like(surface_points) - 0.5) * 0.01      
                                    pp = torch.cat([surface_points, surface_points_neig], dim=0)
                                    g = model.gradient(pp) 
                                    normals_ = g[:, 0, :] / (g[:, 0, :].norm(2, dim=1).unsqueeze(-1) + 10**(-5))
                                    norm_pred = normals_[:N]
                                    diff_norm =  torch.norm(normals_[:N] - normals_[N:], dim=-1)
                                    norm_pred = norm_pred.unsqueeze(0)
                                    #--------------------------------------------------------

                                    #normals SDP are used like normals_gt
                                    normal = normals[i]
                                    pixels_it = pixels[:,i2:i2+n_max_network_queries]
                                    normal_ps = normal[:,pixels_it[1],pixels_it[0]].T
                                    normal_ps = normal_ps[weight_pos]
                                    normal_ps = normal_ps.unsqueeze(0)
                                    
                                    #angle applied to for DiLigenT x because these camera arround Z-axis (y is the view axis in SDPS)
                                    norm_pred = norm_pred[:,torch.abs(normal_ps[0,:,2])>min_norm_angle]
                                    normal_ps = normal_ps[:,torch.abs(normal_ps[0,:,2])>min_norm_angle]
                                    #world_mat = extrinsic[13]   #armadillo.yaml
                                    world_mat = extrinsic_tens[i].unsqueeze(0)
                                    normal_ps = torch.einsum('bij,bnj->bni', world_mat[:,:3,:3]*torch.tensor([[[1,-1,-1]]],dtype=torch.float32).to(device), normal_ps)
                                    #--------------------------------------------------------
                                    #---------------compute loss fonction-----------------
                                    loss_t, loss_occ_in_obj_t, loss_occ_mask_0_t, grad_loss_t, normal_loss_t = loss_fonction(weight_pos, 
                                                occ_sfs[i2:i2+n_max_network_queries].clone(), alpha, diff_norm, norm_pred,normal_ps, it)
                                    
                                    #------- Compute the mean of all forward passes for printing.------------
                                    loss += loss_t
                                    loss_occ_in_obj += loss_occ_in_obj_t
                                    grad_loss += grad_loss_t
                                    loss_occ_mask_0 += loss_occ_mask_0_t
                                    normal_loss += normal_loss_t
                                    
                                    nb+=1
                                    loss_t.backward()
                                    optimizer.step()
                                    
                                    time_iters += time.time() - time_start

                                    if  N>50 and best_error >= normal_loss_t :
                                        best_model = copy.copy(model)
                                        best_error = float(loss)

                                    if(it % 1000 == 0 and it > 0):
                                        print("check point")
                                        checkpoint_io.save(model_file_name+'.pt', epoch_it=epoch_it, it=it,time_all=(time_all+time_iters),
                                                        loss_val_best=loss.item())  
                                    
                                    if(it % 500 == 0 and it > 0):
                                            print('Backup checkpoint')
                                            checkpoint_io.save(model_file_name+'_%d.pt' % it, epoch_it=epoch_it, it=it,time_all=(time_all+time_iters),
                                                        loss_val_best=loss.item())
                                    
                                    
                                    if(it % 100 == 0):
                                        time_all += time_iters
                                        loss /= nb
                                        loss_occ_in_obj /= nb
                                        loss_occ_mask_0 /= nb
                                        grad_loss /= nb
                                        normal_loss /= nb

                                        

                                        print(#'ba:{:03d}'.format(epoch_it),
                                              'It:{:05d}'.format(it), 
                                            #' img:{:02.0f}'.format(float(i)),
                                            ' loss:{:03.4f}'.format(float(loss.item())),
                                            ' best_loss:{:03.4f}'.format(float(best_error)),
                                            ' occ_in:{:03.4f}'.format(float(loss_occ_in_obj.item())),
                                            ' occ_msk:{:03.4f}'.format(float(loss_occ_mask_0.item())),
                                            ' grad:{:03.4f}'.format(float(grad_loss.item())),
                                            ' norm:{:03.4f}'.format(float(normal_loss.item())),
                                            ' time:{:03.2f}'.format(float(time_iters/60.)),
                                            '/{:03.1f} m.'.format(float(time_all/60.)))
                                        
                                        time_iters = 0
                                        loss, loss_occ_mask_0, loss_occ_in_obj,grad_loss,normal_loss, nb = 0,0,0,0,0,0
                                    

                                    # --- The visualise uses the UniSurf (PSNRF) method, so the same configuration is applied. ---
                                    # --- This step is optional, as the 3D object can be generated directly from the grid. ---
                                    # --- All parameters of PSNERF are included directly in the files network.py, rendering_sfs.py, training_sfs.py---
                                    # --- the visualise an other objetc it is necessary to change the depth_range (= torch.tensor([34,36])) in the rendering_sfs.py---
                                    show_ps_mae = True
                                    if(it % 2000 == 0 and it > 0):  
                                            if(show_ps_mae) :    
                                                out_render_path = os.path.join(visualize_path, 'vis_{:06d}.png'.format(it))
                                                print("**PS Normals -> Start visualise with the PS normals.",out_render_path)
                                                render_visdata(
                                                        imgs_test,
                                                        norm_masks_test,
                                                        extrinsic_test,
                                                        K_tens,
                                                        normals_test,
                                                        it, out_render_path,
                                                        best_model,
                                                        device)
                                            
                                            out_render_path = os.path.join(visualize_path, 'vis_gt_{:06d}.png'.format(it))
                                            print("** GT Normal -> Start visualise with the GT normals.",out_render_path)
                                            render_visdata(
                                                    imgs_test,
                                                    norm_masks_test,
                                                    extrinsic_test,
                                                    K_tens,
                                                    normals_gt_test, 
                                                    it, out_render_path,
                                                    best_model,
                                                    device)

                                            best_error = 999999.

                                
        except EOFError as e :
            print(">>>> End of File and repeat <<<<<<")
            continue            

