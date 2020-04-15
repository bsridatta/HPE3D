
import numpy as np
import os
import scipy.io as sio
import h5py

img_path = '/home/datta/lab/HPE_datasets/human36m/'
save_path = './'

h5name = save_path + 'h36m17.h5'

f = h5py.File(h5name, 'r')

idx = 10

for key in f.keys():
    print(key, f[key].shape)
    print(f[key][idx])

f.close()
