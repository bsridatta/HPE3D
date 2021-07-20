import gc
import os
from typing import Dict, List, Optional, Union

import h5py
import numpy as np
import torch
from torch._C import dtype
from torch.utils.data import Dataset, dataset

from processing import preprocess, translate_and_project
from datasets.h36m_utils import H36M_NAMES, ACTION_NAMES
from datasets.skeleton import Skeleton


class Compose(object):
    """Composes several transforms together.
    source: https://github.com/pytorch/vision/blob/8759f3035b860365cae15656a5636e408163e8b4/torchvision/transforms/transforms.py#L58
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pose2d, pose3d=None):
        val = {}
        val["pose2d"] = pose2d
        if pose3d:
            val["pose3d"] = pose3d
        for t in self.transforms:
            val = t(val)
        return val


class Flip(object):
    def __init__(self, flipped_indices: List[str], p: float = 0.5) -> None:
        self.flipped_indices = flipped_indices
        self.p = p

    def __call__(self, val: Dict[str, torch.Tensor]):
        if torch.rand(1) < self.p:
            val["pose2d"] = self.flip(val["pose2d"], self.flipped_indices)
            if "pose3d" in val.keys():
                val["pose3d"] = self.flip(val["pose3d"], self.flipped_indices)
        return val

    @staticmethod
    def flip(val: torch.Tensor, flipped_indices):
        # switch magnitude and direction
        val = val[flipped_indices]
        val[:, 0] *= -1
        return val


class Occlude(object):
    def __init__(self, joints: List[str], p: float = 0.0, n_occlude: int = -1):
        self.p = p
        self.n_occlude = n_occlude
        self.can_occlude = torch.Tensor(
            self.get_joints_to_occlude(joints, select_all=False)
        )

    @staticmethod
    def get_joints_to_occlude(joints: List[str], select_all: bool) -> List[int]:
        can_occlude = [1] * len(joints)
        if select_all:
            return can_occlude
        else:  # select outer joints
            for idx, joint in enumerate(joints):
                if "R_" not in joint and "L_" not in joint:
                    can_occlude[idx] = 0
        return can_occlude

    def __call__(self, val):
        val["mask"] = torch.ones_like(val["pose2d"])
        if torch.rand(1) < self.p:
            if self.n_occlude != -1:
                n_occlude = self.n_occlude
            elif torch.rand(1) < 0.5:
                n_occlude = 1
            else:
                n_occlude = 2

            val["pose2d"], miss_idx = self.random_zero_rows(
                val["pose2d"], self.can_occlude, n_occlude
            )
            val["mask"] = val["mask"][miss_idx, :] = 0
        return val

    @staticmethod
    def random_zero_rows(val: torch.Tensor, weights: torch.Tensor, n_samples: int):
        miss_idx = torch.multinomial(weights, n_samples, replacement=False)
        val[miss_idx, :] = 0
        return val, miss_idx


class H36M(Dataset):
    def __init__(
        self,
        h5_filepath: str,
        is_ss: bool = True,
        is_train: bool = False,
        all_keys: bool = False,
        p_occlude: float = 0.0,
    ):
        """H36M dataset

        Args:
            h5_filepath (str): path to h5 file
            is_ss (bool, optional): [description]. Defaults to True.
            is_train (bool, optional): [description]. Defaults to False.
            all_keys (bool, optional): include all keys. Defaults to False.
            p_occlude (float, optional): emulate missed detection due to occlusion (x,y set to 0). Defaults to 0.0.
        """

        self.is_train = is_train
        self.skel = Skeleton()

        h5 = h5py.File(h5_filepath, "r")

        if all_keys:
            self.keys = h5.keys()
        elif is_train and is_ss:
            self.keys = ["pose2d", "idx"]
        else:
            self.keys = ["pose2d", "pose3d", "idx"]

        print(f"[INFO]: Processing dataset")
        self.data: Dict[str, torch.Tensor] = {}
        for key in self.keys:
            val = h5.get(key)
            if key in ["pose2d", "pose3d"]:
                val = np.array(val)  # preprocessing in numpy is easy
                val = preprocess(val, self.skel.joints, self.skel.root_idx, is_ss=is_ss)
            self.data[key] = torch.tensor(val, dtype=torch.float32)
        h5.close()

        if self.is_train:
            self.transform = Compose(
                [
                    Flip(self.skel.flipped_indices, p=0.5),
                    Occlude(self.skel.joints_15, p=0, n_occlude=-1),
                ]
            )
        else:
            self.transform = Compose(
                [
                    Occlude(self.skel.joints_15, p=p_occlude, n_occlude=-1),
                ]
            )
        
        assert(is_ss or not p_occlude) # no occlusion for supervised

    def __len__(self):
        return len(self.data["idx"])

    def __getitem__(self, idx):
        sample = {}
        for key in self.keys:
            sample[key] = self.data[key][idx]
        sample.update(self.transform(sample["pose2d"], sample.get("pose3d")))
        return sample


if __name__ == "__main__":
    """
    Can be used to get norm stats for all subjects/ # Just for easily access content.
    """
    h5_filepath = f"src/data/h36m_train_sh_ft.h5"
    # image_path = f"{os.getenv('HOME')}/lab/HPE_datasets/h36m_poselifter/"
    dataset = H36M(h5_filepath, is_train=True)
    print("[INFO]: Length of the dataset: ", len(dataset))
    print("[INFO]: One sample -")

    sample = dataset.__getitem__(10)
    for k, v in zip(sample.keys(), sample.values()):
        print(k, v.size(), v.dtype, end="\n")
        pass
