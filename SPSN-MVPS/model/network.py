import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    ''' Network class containing occupanvy and appearance field
    
    Args:
        cfg (dict): network configs
    '''

    def __init__(self, **kwargs):
        super().__init__()
        #cfg = cfg_all['model']
        out_dim = 4
        dim = 3
        self.num_layers = 8
        hidden_size = 512
        self.octaves_pe = 6
        self.octaves_pe_views = 4
        self.skips = [4]
        self.rescale = 1.0
        self.feat_size = 512
        geometric_init = True 

        bias = 0.6

        # init pe
        dim_embed = dim*self.octaves_pe*2 + dim
        dim_embed_view = dim + dim*self.octaves_pe_views*2 + dim + dim + self.feat_size 
        self.transform_points = PositionalEncoding(L=self.octaves_pe)
        self.transform_points_view = PositionalEncoding(L=self.octaves_pe_views)

        ### geo network
        dims_geo = [dim_embed]+ [ hidden_size if i in self.skips else hidden_size for i in range(0, self.num_layers)] + [self.feat_size+1] 
        self.num_layers = len(dims_geo)
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skips:
                out_dim = dims_geo[l + 1] - dims_geo[0]
            else:
                out_dim = dims_geo[l + 1]

            lin = nn.Linear(dims_geo[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims_geo[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif self.octaves_pe > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif self.octaves_pe > 0 and l in self.skips:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_geo[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            
            lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        ## appearance network
        dims_view = [dim_embed_view]+ [ hidden_size for i in range(0, 4)] + [3]

        self.num_layers_app = len(dims_view)

        for l in range(0, self.num_layers_app - 1):
            out_dim = dims_view[l + 1]
            lina = nn.Linear(dims_view[l], out_dim)
            lina = nn.utils.weight_norm(lina)
            setattr(self, "lina" + str(l), lina)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

       
    def infer_occ(self, p):

        pe = self.transform_points(p/self.rescale)
        x = pe
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skips:
                x = torch.cat([x, pe], -1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)     
        return x
    
    
    def gradient(self, p2, tflag=True):
        p = p2/10
        with torch.enable_grad():
            p.requires_grad_(True)
            y = self.infer_occ(p)[...,:1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=tflag,
                retain_graph=tflag,
                only_inputs=True, allow_unused=tflag)[0]
            return gradients.unsqueeze(1)

    def forward(self, p2, ray_d=None, only_occupancy=False, return_logits=False,return_addocc=False, noise=False, **kwargs):
        p = p2/10
        x = self.infer_occ(p)
        if only_occupancy:
            return self.sigmoid(x[...,:1] * -10.0)
        elif return_logits:
            return -1*x[...,:1]


class PositionalEncoding(object):
    def __init__(self, L=10):
        self.L = L
    def __call__(self, p):
        pi = 1.0
        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * pi * p), 
             torch.cos((2 ** i) * pi * p)],
             dim=-1) for i in range(self.L)], dim=-1)
        return torch.cat([p, p_transformed], dim=-1)