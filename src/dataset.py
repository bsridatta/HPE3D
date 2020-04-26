import gc

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class H36M(Dataset):

    def __init__(self, subjects, annotation_file, image_path, no_images=False):
        self.no_images = no_images  # incase of only lifting 2D-3D

        # Data Specific Information - Reference - https://github.com/mks0601/3DMPPE_POSENET_RELEASE/blob/master/data/Human36M/Human36M.py
        self.skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                         (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        self.joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso',
                            'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.flip_pairs = ((1, 4), (2, 5), (3, 6),
                           (14, 11), (15, 12), (16, 13))
        self.root_idx = self.joints_name.index('Pelvis')

        # get labels and metadata including camera parameters
        all_annotations = h5py.File(annotation_file, 'r')
        self.annotations = {}  # only the subjects of interest

        filtered_indices = []
        for i, subject in enumerate(all_annotations['subject']):
            if subject in subjects:
                filtered_indices.append(i)

        for key in all_annotations.keys():
            self.annotations[key] = all_annotations[key][filtered_indices]

        # clear the HDF5 dataset
        all_annotations.close()
        del all_annotations
        gc.collect()

        # to avoid query for every __getitem__
        self.annotation_keys = self.annotations.keys()
        # load image directly in __getitem__
        self.image_path = image_path

    def __len__(self):
        # contains the index of the image files
        return len(self.annotations['idx'])

    def __getitem__(self, idx):
        # Get all data for a sample
        sample = {}
        for key in self.annotation_keys:
            sample[key] = self.annotations[key][idx]
        if not self.no_images:
            image = self.get_image(sample)
            sample['image'] = transforms.ToTensor()(image)
        return sample

    def get_image(self, sample):
        image_dir = 's_%02d_act_%02d_subact_%02d_ca_%02d'\
            % (sample['subject'], sample['action'],
               sample['subaction'], sample['camera'])

        image_file = self.image_path+image_dir+'/' +\
            image_dir+"_"+("%06d" % (sample['idx']))+".jpg"

        image_tmp = Image.open(image_file)
        image = image_tmp.copy()

        # clear PIL
        image_tmp.close()
        del image_tmp
        gc.collect()

        return image


def test_h36m():
    annotation_file = 'data/debug_h36m17.h5'
    image_path = "../../HPE_datasets/h36m/"

    dataset = H36M([1, 5, 6, 7], annotation_file, image_path)
    print("Length of the dataset: ", len(dataset))

    print("One sample -")
    sample = dataset.__getitem__(0)
    for k, v in zip(sample.keys(), sample.values()):
        print(k, v.size, end=" ")
    
    print('\n Without Images')
    
    dataset = H36M([1, 5, 6, 7], annotation_file, image_path, no_images=True)
    print("Length of the dataset: ", len(dataset))

    print("One sample -")
    sample = dataset.__getitem__(0)
    for k, v in zip(sample.keys(), sample.values()):
        print(k, v.size, end=" ")

    print(sample['pose3d_global']-sample['pose3d'])
    del dataset
    del sample
    gc.collect()
    
if __name__ == "__main__":
    test_h36m()
