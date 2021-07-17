import gc
import os
from typing import Optional

import albumentations
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, dataset

from processing import preprocess, project_3d_to_2d
from datasets.h36m_utils import H36M_NAMES, ACTION_NAMES
from datasets.common import COMMON_JOINTS, JOINT_CONNECTIONS


class H36M(Dataset):
    def __init__(
        self,
        h5_filepath: str,
        is_ss: bool = True,
        is_train: bool = False,
    ):
        """H36M dataset

        Args:
            h5_filepath (str): path to h5 file
            is_ss (bool, optional): [description]. Defaults to True.
            is_train (bool, optional): [description]. Defaults to False.
        """
        self.is_train = is_train
        self._set_skeleton_data()
        self.data = {}

        h5 = h5py.File(h5_filepath, "r")
        for key in h5.keys():
            self.data[key] = np.array(h5.get(key))
        h5.close()

        # further process to make the data learnable - zero 3dpose and norm poses
        print(f"[INFO]: processing data samples: {len(self.data['idx'])}")

        self.data = preprocess(self.data, self.joint_names, self.root_idx, is_ss=is_ss)

        # covert data to tensor after preprocessing them as numpys (for convinience)
        for key in self.data.keys():
            self.data[key] = torch.tensor(self.data[key], dtype=torch.float32)

        self.dataset_len = len(self.data["idx"])

        assert self.data["pose2d"].shape[1:] == (15, 2)
        assert self.data["pose3d"].shape[1:] == (15, 3)

    def __len__(self):
        return len(self.data["idx"])

    def __getitem__(self, idx):
        sample = {}
        for key in self.data.keys():
            sample[key] = self.data[key][idx]

        if self.is_train and torch.rand(1) < 0.5:
            sample = self._flip(sample)

        return sample

    def _flip(self, sample):
        # switch magnitude
        sample["pose2d"] = sample["pose2d"][self._flipped_indices]
        sample["pose3d"] = sample["pose3d"][self._flipped_indices]

        # switch direction
        sample["pose2d"][:, 0] *= -1
        sample["pose3d"][:, 0] *= -1

        return sample

    def _set_skeleton_data(self):
        self.joint_names = COMMON_JOINTS.copy()
        self.action_names = list(ACTION_NAMES.values())
        self.root_idx = self.joint_names.index("Pelvis")

        # without pelvis as its removed in the preprocessing step after zeroing
        joints_15 = self.joint_names.copy()
        joints_15.remove("Pelvis")

        self._flipped_indices = []
        for idx, i in enumerate(joints_15):
            if "R_" in i:
                self._flipped_indices.append(joints_15.index(i.replace("R_", "L_")))
            elif "L_" in i:
                self._flipped_indices.append(joints_15.index(i.replace("L_", "R_")))
            else:
                self._flipped_indices.append(idx)


def check_data():
    """
    Can be used to get norm stats for all subjects/ # Just for easily access content.
    """

    h5_filepath = f"src/data/h36m_test_gt_2d.h5"
    # image_path = f"{os.getenv('HOME')}/lab/HPE_datasets/h36m_poselifter/"

    dataset = H36M(h5_filepath, is_train=True)

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
