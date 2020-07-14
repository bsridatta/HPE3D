
import torch.utils.data as data
from mpiinf_dataset import MPIINF
import os
import sys
import gc

sys.path.append("..")
from dataset import H36M

class H36M_MPII(data.Dataset):
    def __init__(self, subjects, annotation_file, image_path, no_images=False, device='cpu', annotation_path=None, train=False):
        self.H36M = H36M(subjects, annotation_file,
                   image_path, no_images, device, annotation_path, train)
        self.MPII = MPIINF(train)
        self.num_h36m = len(self.H36M)
        self.num_mpii = len(self.MPII)
        print('Load %d H36M and %d MPII samples' % (self.num_h36m, self.num_mpii))

    def __getitem__(self, index):
        print(index)
        if index < self.num_mpii:
            return self.MPII[index]
        else:
            return self.H36M[index - self.num_mpii]

    def __len__(self):
        return self.num_h36m + self.num_mpii

def test_mpiinf():
    '''
    Can be used to get norm stats for all subjects
    '''

    annotation_file = f'h36m17'
    # TODO Need to fix image path to work for both datasets
    image_path = f"{os.getenv('HOME')}/lab/HPE_datasets/h36m"

    dataset = H36M_MPII([1, 5, 6, 7, 8],
                   annotation_file, image_path, train=True, no_images=True)

    print("[INFO]: Length of the dataset: ", len(dataset))
    print("[INFO]: One sample -")

    sample = dataset.__getitem__(400000)

    for k, v in zip(sample.keys(), sample.values()):
        print(k, v.size(), v.dtype, end="\t")
        pass

    import viz
    import numpy as np

    print(sample['pose2d'])
    print(sample['pose3d'])

    # viz.plot_pose(pose2d=sample['pose2d'],
    #               pose3d=sample['pose3d'])
    del dataset
    del sample
    gc.collect()


if __name__ == "__main__":
    test_mpiinf()