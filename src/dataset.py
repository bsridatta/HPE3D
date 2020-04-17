import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class H36M(Dataset):

    def __init__(self, protocol, annotation_file, image_path):

        # Data Specific Information - Reference - https://github.com/mks0601/3DMPPE_POSENET_RELEASE/blob/master/data/Human36M/Human36M.py
        self.skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                         (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        self.joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso',
                            'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.flip_pairs = ((1, 4), (2, 5), (3, 6),
                           (14, 11), (15, 12), (16, 13))
        self.root_idx = self.joints_name.index('Pelvis')

        # standard training protocols for Human3.6m
        protocols = {
            1: {"train": [1, 5, 6, 7, 8], "test": [9, 11], "rigid_transform": False},
            2: {"train": [1, 5, 6, 7, 8], "test": [9, 11], "rigid_transform": True}
        }
        self.protocol = protocols[protocol]

        # get labels and metadata including camera parameters
        self.annotation = h5py.File(annotation_file, 'r')
        self.annotation_keys = self.annotation.keys()
        # load image directly in get item
        self.image_path = image_path


    def __len__(self):
        # contains the index of the image files
        return len(self.annotation['idx'])

    def __getitem__(self, idx):
        # Get all data for a sample
        sample = {}

        for key in self.annotation_keys:
            sample[key] = self.annotation[key][idx]

        sample['image'] = self.get_image(sample)

        return sample

    def get_image(self, sample):
        image_dir = 's_%02d_act_%02d_subact_%02d_ca_%02d'\
            % (sample['subject'], sample['action'],\
            sample['subaction'], sample['camera'])

        image_file = self.image_path+image_dir+'/'+\
                        image_dir+"_"+("%06d"%(sample['idx']))+".jpg"
        
        return Image.open(image_file)

def test_h36m():
    annotation_file = 'data/debug_h36m17.h5'
    image_path = "../../HPE_datasets/h36m/"

    dataset = H36M(1, annotation_file, image_path)    
    print(len(dataset))

    sample = dataset.__getitem__(2)
    for k, v in zip(sample.keys(), sample.values()):
        print(k, v.size)

if __name__ == "__main__":
    test_h36m()