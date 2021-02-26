import gc
import os
from typing import Optional

import albumentations
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, dataset

from src.processing import preprocess, project_3d_to_2d
from src.datasets.h36m_utils import H36M_NAMES, ACTION_NAMES
from src.datasets.common import COMMON_JOINTS, JOINT_CONNECTIONS


class H36M(Dataset):
    def __init__(self, data_file: str, image_path: Optional[str] = None,
                 train: bool = False, projection: bool = True):
        """[summary]

        Args:
            data_file (str): [description]
            image_path (Optional[str], optional): [description]. Defaults to None.
            device (str, optional): [description]. Defaults to 'cpu'.
            train (bool, optional): [description]. Defaults to False.
            projection (bool, optional): [description]. Defaults to True.
        """
        self._train = train
        self._image_path = image_path  # load image directly in __getitem__
        self._set_skeleton_data()
        self._data = {}

        h5 = h5py.File(data_file, 'r')
        for key in h5.keys():
            self._data[key] = np.array(h5.get(key))
        h5.close()

        # further process to make the data learnable - zero 3dpose and norm poses
        print(f"[INFO]: processing data samples: {len(self._data['idx'])}")
        
        self._data = preprocess(
            self._data, self.joint_names, self.root_idx, projection=projection)

        # covert data to tensor after preprocessing them as numpys (messy with tensors)
        for key in self._data.keys():
            self._data[key] = torch.tensor(
                self._data[key], dtype=torch.float32)

        if image_path:
            # Image normalization
            self.augmentations = albumentations.Compose([
                albumentations.Normalize(always_apply=True)
            ])

        self.dataset_len = len(self._data['idx'])

        assert self._data['pose2d'].shape[1:] == (15, 2)
        assert self._data['pose3d'].shape[1:] == (15, 3)

    def __len__(self):
        return len(self._data['idx'])

    def __getitem__(self, idx):
        sample = {}
        for key in self._data.keys():
            sample[key] = self._data[key][idx]

        if self._image_path:
            image = self.get_image_tensor(sample)
            sample['image'] = image

        if self._train and torch.rand(1) < 0:
            sample = self._flip(sample)

        return sample

    def get_image_tensor(self, sample):
        seq_dir = 's_%02d_act_%02d_subact_%02d_ca_%02d'\
            % (sample['subject'], sample['action'],
               sample['subaction'], sample['camera'])

        image_ = Image.open(self._image_path +
                            seq_dir+'/' +
                            seq_dir+"_"+("%06d" % (sample['idx']))+".jpg")

        image = np.array(image_)
        image = self.augmentations(image=image)['image']
        image = np.transpose(image, (2, 0, 1)).astype('float32')
        image = torch.tensor(image, dtype=torch.float32)

        return image

    def _flip(self, sample):
        # switch magnitude
        sample['pose2d'] = sample['pose2d'][self._flipped_indices]
        sample['pose3d'] = sample['pose3d'][self._flipped_indices]

        # switch direction
        sample['pose2d'][:, 0] *= -1
        sample['pose3d'][:, 0] *= -1

        # TODO add image flipping with albumentaitons

        return sample

    def _set_skeleton_data(self):
        self.joint_names = COMMON_JOINTS.copy()
        self.action_names = list(ACTION_NAMES.values())
        self.root_idx = self.joint_names.index('Pelvis')

        # without pelvis as its removed in the preprocessing step after zeroing
        joints_15 = self.joint_names.copy()
        joints_15.remove('Pelvis')

        self._flipped_indices = []
        for idx, i in enumerate(joints_15):
            if "R_" in i:
                self._flipped_indices.append(
                    joints_15.index(i.replace("R_", "L_")))
            elif "L_" in i:
                self._flipped_indices.append(
                    joints_15.index(i.replace("L_", "R_")))
            else:
                self._flipped_indices.append(idx)

# Just for easily access content


def check_data():
    '''
    Can be used to get norm stats for all subjects
    '''

    data_file = f'src/data/h36m_test_gt_2d.h5'
    image_path = f"{os.getenv('HOME')}/lab/HPE_datasets/h36m_poselifter/"

    dataset = H36M(data_file, train=True)

    print("[INFO]: Length of the dataset: ", len(dataset))
    print("[INFO]: One sample -")

    sample = dataset.__getitem__(10)

    for k, v in zip(sample.keys(), sample.values()):
        print(k, v.size(), v.dtype, end="\n")
        pass

    del dataset
    del sample
    gc.collect()


if __name__ == "__main__":
    check_data()
