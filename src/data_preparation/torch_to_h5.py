# %%
import torch
import os
import h5py

path = f'{os.getenv("HOME")}/lab/HPE_datasets/torch_pickles/'
pickle = torch.load(path+"test_2d_ft.pth.tar")

# %%
print(pickle)