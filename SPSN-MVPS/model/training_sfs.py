''' From PSNERF Code : https://github.com/ywq/psnerf'''

import torch

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from model.rendering_sfs import unisurf



to_img = lambda x: (x.astype(np.float32).clip(0,1) * 255).astype(np.uint8)
to_numpy = lambda x: x.detach().cpu().numpy()
to_hw = lambda x, h, w: x.reshape(w,h,-1).permute(1,0,2)

from utils.tools import MAE
cm = plt.get_cmap('jet')



def render_visdata(imgs,masks,world_mats,camera_mat,normals, it, out_render_path, model, device):
        #self.model.eval()   <-----------------
        save_img = []
        MAE_ALL =0.
        MAE_ALL3 =0.
        camera_mat = camera_mat.unsqueeze(0)
        scale_mat = torch.eye(4).unsqueeze(0).to(device)

        #for di, data in enumerate(data_loader):
        for di in range (imgs.shape[0]):
            #if di!=3 : continue

            img =imgs[di].permute(2,0,1).unsqueeze(0)
            mask = masks[di].unsqueeze(0).unsqueeze(0)
            world_mat = world_mats[di].unsqueeze(0)
            
            normal =normals[di].unsqueeze(0)
            
                    
            #(img2, mask2, world_mat2, camera_mat2, scale_mat2, img_idx, normal2, norm_mask, mask_valid) = \
            #    process_data_dict(data,device)

            
            h, w = img.shape[-2:] #resolution
            p_loc, pixels = arange_pixels(resolution=(h, w))
            pixels = pixels.to(device)
            ploc = p_loc.to(device)

            img_iter = [to_img(to_numpy(img))[0].transpose(1,2,0)]

            with torch.no_grad():
                norm_pred,mask_pred = [], []
                for ii, pixels_i in enumerate(torch.split(ploc, 1024, dim=1)):
                    #if(ii%50==0) : print(di+1,"/",imgs.shape[0], ") -> visdata -> ",ii," / ",(ploc.shape[1]/1024))
                    out_dict = unisurf(model,device,
                                    pixels_i, camera_mat, world_mat, scale_mat, it=it)
                    norm_pred.append(out_dict['normal_pred'])
                    mask_pred.append(out_dict['mask_pred'])
                mask_pred = to_numpy(to_hw(torch.cat(mask_pred, dim=0),h,w)).repeat(3,axis=-1)
                # normal
                norm_pred = to_numpy(to_hw(torch.cat(norm_pred, dim=1),h,w))
                norm_pred = np.einsum('ij,hwi->hwj', to_numpy(world_mat)[0,:3,:3]*np.array([[1,-1,-1]]),norm_pred)
                norm_pred = norm_pred.clip(-1,1)
                norm_pred[norm_pred[:,:,2]<0] = -norm_pred[norm_pred[:,:,2]<0]
                img_iter.append(to_img(norm_pred/2.+0.5))
                if normal is not None:

            

                    mask_2 = to_numpy(mask.bool())[0,0] #& mask_pred[...,0]
                    img_iter.append(to_img(to_numpy(normal[0].permute(1,2,0))/2.+0.5))
                    error = MAE(norm_pred.clip(-1,1),to_numpy(normal[0].permute(1,2,0)).clip(-1,1))[1]/57
                    error_MAE = MAE(norm_pred,to_numpy(normal[0].permute(1,2,0)).clip(-1,1),mask_2)[0]
                    mask_3 = to_numpy(mask.bool())[0,0] & mask_pred[...,0]
                    error_MAE3 = MAE(norm_pred,to_numpy(normal[0].permute(1,2,0)).clip(-1,1),mask_3)[0]
                    print(di+1,"/",imgs.shape[0], "): MAE",error_MAE,", MAE_intersect",error_MAE3)
                    MAE_ALL += error_MAE
                    MAE_ALL3 += error_MAE3
                    
                    img_iter.append(to_img(cm(error.clip(0,1)*(to_numpy(mask.bool())[0,0]|mask_pred[...,0]))[...,:3]))
                # mask
                img_iter.append(to_img(mask_pred))
                
            
            save_img.append(np.concatenate(img_iter, axis=-2))
        save_img = np.concatenate(save_img, axis=0)
        save_img = Image.fromarray(save_img.astype(np.uint8)).convert("RGB")
        save_img.save(out_render_path)
        save_img.save(out_render_path + '_MAE_{:03.2f}.png'.format(MAE_ALL/5.))
        model.train()
        print('MAE_ALL = {:03.5f})'.format(MAE_ALL/5.),', MAE_ALL3 = {:03.5f})'.format(MAE_ALL3/5.))
        return 

def process_data_dict(data, device, normal_loss=True):
        ''' Processes the data dictionary and returns respective tensors

        Args:
            data (dictionary): data dictionary
        '''
        
        img = data.get('img').to(device)
        img_idx = data.get('img.idx')
        batch_size, _, h, w = img.shape
        mask_img = data.get('img.mask', torch.ones(batch_size, h, w)).unsqueeze(1).to(device)
        world_mat = data.get('img.world_mat').to(device)
        camera_mat = data.get('img.camera_mat').to(device)
        scale_mat = data.get('img.scale_mat').to(device)
        normal = data.get('img.normal').to(device) if normal_loss else None
        norm_mask = data.get('img.norm_mask').unsqueeze(1).to(device) if normal_loss else None
        mask_valid = data.get('img.mask_valid', torch.ones(batch_size, h, w)).unsqueeze(1).to(device)

        return (img, mask_img, world_mat, camera_mat, scale_mat, img_idx, normal, norm_mask, mask_valid)


def arange_pixels(resolution=(128, 128), batch_size=1, image_range=(-1., 1.),
                  subsample_to=None):
    ''' Arranges pixels for given resolution in range image_range.

    The function returns the unscaled pixel locations as integers and the
    scaled float values.

    Args:
        resolution (tuple): image resolution
        batch_size (int): batch size
        image_range (tuple): range of output points (default [-1, 1])
        subsample_to (int): if integer and > 0, the points are randomly
            subsampled to this value
    '''
    h, w = resolution
    n_points = resolution[0] * resolution[1]

    # Arrange pixel location in scale resolution
    pixel_locations = torch.meshgrid(torch.arange(0, w), torch.arange(0, h))
    pixel_locations = torch.stack(
        [pixel_locations[0], pixel_locations[1]],
        dim=-1).long().view(1, -1, 2).repeat(batch_size, 1, 1)
    pixel_scaled = pixel_locations.clone().float()

    # Shift and scale points to match image_range
    scale = (image_range[1] - image_range[0])
    loc = (image_range[1] - image_range[0])/ 2
    pixel_scaled[:, :, 0] = scale * pixel_scaled[:, :, 0] / (w - 1) - loc
    pixel_scaled[:, :, 1] = scale * pixel_scaled[:, :, 1] / (h - 1) - loc

    # Subsample points if subsample_to is not None and > 0
    if (subsample_to is not None and subsample_to > 0 and
            subsample_to < n_points):
        idx = np.random.choice(pixel_scaled.shape[1], size=(subsample_to,),
                               replace=False)
        pixel_scaled = pixel_scaled[:, idx]
        pixel_locations = pixel_locations[:, idx]

    return pixel_locations, pixel_scaled