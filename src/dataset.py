import gc
import logging
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
        all_annotations = h5py.File(f'{annotation_file}', 'r')
        self.annotations = {}  # to store only the subjects of interest

        # get indices of subjects of interest and filter them
        filtered_indices = []
        for i, subject in enumerate(all_annotations['subject']):
            if subject in subjects:
                filtered_indices.append(i)

        for key in all_annotations.keys():
            self.annotations[key] = all_annotations[key][filtered_indices]

        # further process to make the data learnable - zero3d and norm poses
        logging.info(f'processing subjects: {subjects}')
        self.annotations, norm_stats = preprocess(
            self.annotations, self.root_idx)

        for key in self.annotations:
            self.annotations[key] = torch.tensor(self.annotations[key], dtype=torch.float32, device=self.device)

        # save norm_stats to denormalize data for evaluation
        f = h5py.File(
            f"{os.path.dirname(os.path.abspath(__file__))}/data/norm_stats.h5", 'w')
        for key in norm_stats.keys():
            f[key] = norm_stats[key]

        # clear the HDF5 datasets
        f.close()
        del f
        all_annotations.close()
        del all_annotations
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
            sample[key] = self.annotations[key][idx]
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
        # print("org max ", np.max(image), image.shape)
        image = self.augmentations(image=image)['image']
        # print("aug max ", np.max(image), image.shape)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float32, device=self.device)

        # *Note* - toTensor converts HWC to CHW so no need NOT to do explicitly
        # But if you do torch.tensor() you have to do it manually

        # clear PIL
        image_tmp.close()
        del image_tmp
        gc.collect()

        return image


'''
test function for sanity check only - ignore
'''


def test_h36m():
    annotation_file = f'{os.path.dirname(os.path.abspath(__file__))}/data/debug_h36m17.h5'
    image_path = f"/home/datta/lab/HPE_datasets/h36m/"

    dataset = H36M([1, 5, 6], annotation_file, image_path)
    print("Length of the dataset: ", len(dataset))

    print("One sample -")
    sample = dataset.__getitem__(0)
    for k, v in zip(sample.keys(), sample.values()):
        print(k, v.size(), v.dtype, end="\t")
    print("")

    print(sample['pose2d'], '\n\n\n')
    print(sample['pose3d'])

    del dataset
    del sample
    gc.collect()


if __name__ == "__main__":
    test_h36m()
