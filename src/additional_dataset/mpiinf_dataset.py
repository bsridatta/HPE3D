import torch
from h5py import File
from torch.utils.data import Dataset
import os
import gc
import sys
import numpy as np
import pdb

sys.path.insert(
    0, f'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}')
from processing import preprocess

'''Code adapted from dataset provided by https://github.com/juyongchang/PoseLifter
'''


class MPIINF(Dataset):
    def __init__(self, train):
        if train:
            self.split = "train"
        else:
            self.split = "val"

        self.annotations = {}
        # More @ print([key for key in f.keys()])
        tags = ['pose2d', 'pose3d', 'subject', 'bbox']

        f = File('%s/inf_%s.h5' %
                 (f"{os.getenv('HOME')}/lab/HPE_datasets/annot/inf", self.split), 'r')

        for tag in tags:
            self.annotations[tag] = f[tag][:]
        f.close()

        self.annotations['action'] = self.annotations['subject']
        self.annotations['camera'] = self.annotations['subject']
        self.annotations['idx'] = self.annotations['subject']
        self.annotations['subaction'] = self.annotations['subject']
    

        self.flipped_indices = [3, 4, 5, 0, 1, 2,
                                6, 7, 8, 9, 13, 14, 15, 10, 11, 12]
        self.annotation_keys = self.annotations.keys()
        self.root_idx = 0

        self.annotations = preprocess(
            self.annotations, self.root_idx, normalize_pose=False)

        # print(self.annotations['pose2d'].mean(axis=(0, 2)))
        # print(self.annotations['pose3d'].mean(axis=(0, 2)))

        # covert data to tensor after preprocessing them as numpys (hard with tensors)
        for key in self.annotation_keys:
            self.annotations[key] = torch.tensor(
                self.annotations[key], dtype=torch.float32)

    def __len__(self):
        return self.annotations['pose2d'].shape[0]

    def __getitem__(self, idx):
        sample = {}
        for key in self.annotation_keys:
            sample[key] = self.annotations[key][idx]
       # Augmentation - Flip
        if self.split == "train" and np.random.random() < 0.2:
            sample = self.flip(sample)
        return sample

    def flip(self, sample):
        sample['pose2d'] = sample['pose2d'][self.flipped_indices]
        sample['pose3d'] = sample['pose3d'][self.flipped_indices]
        sample['pose2d'][:, 0] *= -1
        sample['pose3d'][:, 0] *= -1
        return sample


def test_mpiinf():
    '''
    Can be used to get norm stats for all subjects
    '''

    annotation_file = f'h36m17'
    image_path = f"{os.getenv('HOME')}/lab/HPE_datasets/inf"

    dataset = MPIINF('train')

    print("[INFO]: Length of the dataset: ", len(dataset))
    print("[INFO]: One sample -")

    sample = dataset.__getitem__(50)

    for k, v in zip(sample.keys(), sample.values()):
        print(k, v.size(), v.dtype, end="\t")
        pass

    import viz
    import numpy as np

    print(sample['pose2d'])
    print(sample['pose3d'])

    viz.plot_pose(pose2d=sample['pose2d'],
                  pose3d=sample['pose3d'])
    del dataset
    del sample
    gc.collect()


if __name__ == "__main__":
    test_mpiinf()
