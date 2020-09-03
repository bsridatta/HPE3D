import os
import h5py
from src.processing import post_process
import torch

def speed():
    device = 'cuda'
    recon = torch.rand((100000,16,3), device=device, dtype=torch.float32)
    target = torch.rand((100000,16,3), device=device, dtype=torch.float32)
    scale = torch.rand((100000,1), device=device, dtype=torch.float32)

    recon, target = post_process(
        recon.to('cpu'), target.to('cpu'), scale.to('cpu'), self_supervised=True,procrustes_enabled=True)
    

if __name__ == '__main__':

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        speed()
    print(prof)
    