import gc
import os

import albumentations
import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from processing import preprocess
import time

class H36M(Dataset):

    def __init__(self, subjects, annotation_file, image_path, no_images=False, device='cpu'):
        self.no_images = no_images  # incase of only lifting 2D-3D
        self.device = device
        # Data Specific Information
        # Reference - https://github.com/mks0601/3DMPPE_POSENET_RELEASE/blob/master/data/Human36M/Human36M.py

        self.skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                         (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        self.joints_name = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso',
                            'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.flip_pairs = ((1, 4), (2, 5), (3, 6),
                           (14, 11), (15, 12), (16, 13))
        self.root_idx = self.joints_name.index('Pelvis')

        # get labels and metadata including camera parameters
        subj_name = "".join(str(sub) for sub in subjects)
        annotations_h5 = h5py.File(f'{os.path.dirname(os.path.abspath(__file__))}/data/{annotation_file}_{subj_name}.h5', 'r')

        self.annotations = {}  # to store only the subjects of interest
        for key in annotations_h5.keys():
            self.annotations[key] = annotations_h5[key][:]

        # further process to make the data learnable - zero3d and norm poses
        print(f'[INFO]: processing subjects: {subjects}')
        self.annotations, norm_stats = preprocess(
            self.annotations, self.root_idx)


        # save norm_stats to denormalize data for evaluation
        subj_name = "".join(str(sub) for sub in subjects)
        norm_stats_name = f"norm_stats_{annotation_file}_{subj_name}.h5"
        f = h5py.File(
            f"{os.path.dirname(os.path.abspath(__file__))}/data/{norm_stats_name}", 'w')

        for key in norm_stats.keys():
            f[key] = norm_stats[key]

        # clear the HDF5 datasets
        annotations_h5.close()
        f.close()
        del annotations_h5
        del f
        gc.collect()

        # to avoid query for every __getitem__
        self.annotation_keys = self.annotations.keys()
        # load image directly in __getitem__
        self.image_path = image_path

        self.augmentations = albumentations.Compose([
            albumentations.Normalize(always_apply=True)
        ])

    def __len__(self):
        # contains the index of the image files
        return len(self.annotations['idx'])

    def __getitem__(self, idx):
        # Get all data for a sample
        sample = {}
        for key in self.annotation_keys:
            sample[key] = torch.tensor(
                self.annotations[key][idx], dtype=torch.float32)
        if not self.no_images:
            image = self.get_image_tensor(sample)
            sample['image'] = image
        return sample

    def get_image_tensor(self, sample):
        image_dir = 's_%02d_act_%02d_subact_%02d_ca_%02d'\
            % (sample['subject'], sample['action'],
               sample['subaction'], sample['camera'])

        image_file = self.image_path+image_dir+'/' +\
            image_dir+"_"+("%06d" % (sample['idx']))+".jpg"

        image_tmp = Image.open(image_file)
        image = image_tmp.copy()
        image = np.array(image)
        image = self.augmentations(image=image)['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float32)

        # *Note* - toTensor converts HWC to CHW so no need NOT to do explicitly
        # But if you do torch.tensor() you have to do it manually

        # clear PIL
        image_tmp.close()
        del image_tmp
        gc.collect()

        return image


'''
Can be used to get norm stats for all subjects
'''


def test_h36m():
    annotation_file = f'h36m17'
    image_path = f"/home/datta/lab/HPE_datasets/h36m/"
    
    
    dataset = H36M([1,5,6,7,8,9,11], annotation_file, image_path)
    
    print("[INFO]: Length of the dataset: ", len(dataset))
    print("[INFO]: One sample -")
    
    sample = dataset.__getitem__(0)

    
    for k, v in zip(sample.keys(), sample.values()):
        print(k, v.size(), v.dtype, end="\t")
        pass

    # print(sample['pose2d'], '\n\n\n')
    # print(sample['pose3d'])

    del dataset
    del sample
    gc.collect()


if __name__ == "__main__":
    test_h36m()
