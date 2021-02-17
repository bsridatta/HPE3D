# %%
from collections import defaultdict
from typing import Dict, Union, Tuple, List
from numpy.core.arrayprint import SubArrayFormat
import torch
import os
import h5py
import numpy as np
from src.data_preparation.get_metadata import action_to_id, camera_id_to_num

data_path: str = f'{os.getenv("HOME")}/lab/HPE_datasets/gt_sh/'

TRAIN: bool = True

if TRAIN:
    subject_list = [1, 5, 6, 7, 8]
    skip_frames = 5
else:
    subject_list = [9, 11]
    skip_frames = 64

subjects_str = "".join(str(x) for x in subject_list)

h5name = f'h36m17_' + subjects_str


# %%