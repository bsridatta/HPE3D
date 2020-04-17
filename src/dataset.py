'''
reference code - https://github.com/mks0601/3DMPPE_POSENET_RELEASE/blob/master/data/Human36M/Human36M.py
'''
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class H36M(Dataset):
    def __init__(self, protocol, annot_path, data_path):
        self.protocol = protocol

        self.skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
        self.joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.flip_pairs = ((1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13))
        self.root_idx = self.joints_name.index('Pelvis')


