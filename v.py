import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import collections as coll
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import gc
import numpy as np
import torch as pt
import torch
import torch.nn as nn
import torch

def print_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated() / 1024 ** 3
    cached_memory = torch.cuda.memory_reserved() / 1024 ** 3
    print(f"Allocated GPU Memory: {allocated_memory:.2f} GB")
    print(f"Cached GPU Memory: {cached_memory:.2f} GB")

print_gpu_memory()
import time as TIME
import torch_tensorrt

from FHNCUDAlib import FHNCUDA
import numpy as np
import torch 

loaded_module = nn.Linear(10,10)
#loaded_module.load_state_dict()

#print(loaded_module.state_dict())
st=torch.load('network.0.pth')
#print(st)


pt.set_grad_enabled (False) 
numinputs=1
numoutputs=2
class Net(nn.Module):
    def __init__(self, numinputs, numoutputs, numlayers=4, H=10):
        super(Net, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.utils.weight_norm(nn.Linear(numinputs, H), name='weight', dim=0).cuda())

        for _ in range(numlayers - 1):
            self.layers.append(nn.utils.weight_norm(nn.Linear(H, H), name='weight', dim=0).cuda())

        self.final_layer = nn.Linear(H, numoutputs).cuda()

        for layer in self.layers:
            layer.eval()
        self.final_layer.eval()

    def forward(self, x):
        for layer in self.layers:
            x = f.silu(layer(x))

        return self.final_layer(x)

    def load(self, od):
        for k, v in od.items():
            if k.startswith('_impl.layers'):
                layer_num = int(k.split('.')[2])
                layer = self.layers[layer_num]
                if k.endswith('linear.weight'):
                    layer.weight_v.data = v
                    layer.weight_v.requires_grad = False
                elif k.endswith('linear.weight_g'):
                    layer.weight_g.data = v
                    layer.weight_g.requires_grad = False
                elif k.endswith('linear.bias'):
                    layer.bias.data = v
                    layer.bias.requires_grad = False
            elif k == '_impl.final_layer.linear.weight':
                self.final_layer.weight.data = v
                self.final_layer.weight.requires_grad = False
            elif k == '_impl.final_layer.linear.bias':
                self.final_layer.bias.data = v
                self.final_layer.bias.requires_grad = False

    def __prepare_scriptable__(self):
        for layer in self.layers:
            for hook in layer._forward_pre_hooks.values():
                if hook.__module__ == "torch.nn.utils.weight_norm" and hook.__class__.__name__ == "WeightNorm":
                    torch.nn.utils.remove_weight_norm(layer)
        return self

import itertools



def Modelrun_s(x,M):
   
    gc.collect()
    torch.cuda.empty_cache()
    my2dspace = pt.tensor(x, requires_grad=False).float().cuda()
    M.eval()
    start_time = TIME.time()
    myOutput = M(my2dspace)
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    reftime = TIME.time()- start_time
    myCPUOutput = myOutput.cpu()


    uu = myCPUOutput.numpy()

    #print('uu: ', uu.T[0])

    myCPUOutput.squeeze().detach().numpy()
    gc.collect()
    torch.cuda.empty_cache()
    return uu,reftime


def Modelrun(x, M, batch_size=10*20*10*200):

    my2dspace = x
    M.eval()
        
    num_samples = my2dspace.shape[0]
  
    gc.collect()
    torch.cuda.empty_cache()
    uu_list = []
   

    reftime =0
    
    for i in range(0, num_samples, batch_size):
        print("batch ",i/batch_size)
        print_gpu_memory()

        gc.collect()
        torch.cuda.empty_cache()
        batch_input =torch.tensor(my2dspace[i:i+batch_size], requires_grad=False).float().cuda() 
        print_gpu_memory()

        start_time = TIME.time()
        

        batch_output = M(batch_input)
        torch.cuda.synchronize()
        print_gpu_memory()

        torch.cuda.synchronize()  # Wait for the events to be recorded!
        reftime =reftime+ TIME.time() - start_time
        uu_list.append(batch_output.cpu().numpy())
        del batch_input
        del batch_output
        #print_gpu_memory()
       
 
    uu = np.concatenate(uu_list)
    
    return uu, reftime


