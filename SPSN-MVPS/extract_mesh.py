import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import time
import torch

from model.checkpoints import CheckpointIO
from model.network import NeuralNetwork
from model.extracting import Extractor3D
from SFS_files.mesh_common import *


torch.manual_seed(0)


os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")



obj_name = 'cow'
expname = "test1"
out_dir = '../out'
checkpoint_file = 'model_spsn.pt'
resolution_3D=512
upsampling_steps_3D=1



test_out_path = os.path.join(out_dir,obj_name, expname)
os.makedirs(test_out_path,exist_ok=True)

# Model
model = NeuralNetwork()
out_dir = os.path.join(out_dir, obj_name, expname)
path = os.path.join(out_dir,'models')
checkpoint_io = CheckpointIO(os.path.join(out_dir,'models'), model=model)
checkpoint_io.load(checkpoint_file)

# Generator
generator = Extractor3D(
    model, resolution0=resolution_3D, 
    upsampling_steps=upsampling_steps_3D, 
    device=device
)
# Generate
model.eval()

try:
    t0 = time.time()
    out = generator.generate_mesh(mask_loader=None,clip=False)

    try:
        mesh, stats_dict = out
    except TypeError:
        mesh, stats_dict = out, {}

    

    mesh_out_file = os.path.join(
        test_out_path, obj_name+'_model_odj.obj')
    print("==============================")
    print(mesh_out_file)
    mesh.export(mesh_out_file)
    #os.startfile(mesh_out_file)

except RuntimeError:
    print("Error generating mesh")

