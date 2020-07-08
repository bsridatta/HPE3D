
import torch
from h5py import File
from torch.utils.data import Dataset
import os
import gc 

import numpy as np
import pdb

'''Code adapted from dataset provided by https://github.com/juyongchang/PoseLifter
'''

flip_index = [0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13]

class MPIINF(Dataset):
    def __init__(self, split):
        print('==> Initializing MPI_INF %s data' % (split))

        annot = {}
        tags = ['pose2d', 'pose3d', 'bbox', 'cam_f',
                'cam_c', 'subject', 'sequence', 'video']

        f = File('%s/inf_%s.h5' % (f"{os.getenv('HOME')}/lab/HPE_datasets/annot/inf", split), 'r')

        for tag in tags:
            annot[tag] = np.asarray(f[tag]).copy()
        f.close()

        self.split = split
        self.annot = annot
        self.num_samples = self.annot['pose2d'].shape[0]

        # image size
        self.width = 2048 * 0.5
        self.height = 2048 * 0.5

        print('Load %d MPI_INF %s samples' % (self.num_samples, self.split))

    def get_part_info(self, index):
        pose2d = self.annot['pose2d'][index].copy()
        bbox = self.annot['bbox'][index].copy()
        pose3d = self.annot['pose3d'][index].copy()
        cam_f = self.annot['cam_f'][index].copy()
        cam_c = self.annot['cam_c'][index].copy()
        return pose2d, bbox, pose3d, cam_f, cam_c

    def __getitem__(self, index):
        # get 2d/3d pose, bounding box, camera information
        pose2d, bbox, pose3d, cam_f, cam_c = self.get_part_info(index)
        cam_f = cam_f.astype(np.float32)
        cam_c = cam_c.astype(np.float32)

        # SCALING
        pose2d = pose2d * 0.5
        bbox = bbox * 0.5
        cam_f = cam_f * 0.5
        cam_c = cam_c * 0.5

        # data augmentation (flipping)
        if self.split == 'train' and np.random.random() < 0.5:
            pose2d_flip = pose2d.copy()
            for i in range(len(flip_index)):
                pose2d_flip[i] = pose2d[flip_index[i]].copy()
            pose3d_flip = pose3d.copy()
            for i in range(len(flip_index)):
                pose3d_flip[i] = pose3d[flip_index[i]].copy()
            pose2d = pose2d_flip.copy()
            pose3d = pose3d_flip.copy()
            pose2d[:, 0] = self.width - pose2d[:, 0]
            pose3d[:, 0] *= -1

            #bbox[0] = self.width - bbox[0] - bbox[2]
            #cam_c[0] = self.width - cam_c[0]



        # set data
        data = {'pose2d': pose2d, 'bbox': bbox,
                'pose3d': pose3d,
                'cam_f': cam_f, 'cam_c': cam_c
                }

        return data

    def __len__(self):
        return self.num_samples



def test_mpiinf():
    '''
    Can be used to get norm stats for all subjects
    '''

    annotation_file = f'h36m17'
    image_path = f"{os.getenv('HOME')}/lab/HPE_datasets/inf"

    dataset = MPIINF('train')

    print("[INFO]: Length of the dataset: ", len(dataset))
    print("[INFO]: One sample -")

    sample = dataset.__getitem__(5)

    print(sample) 

    exit()
    for k, v in zip(sample.keys(), sample.values()):
        print(k, v.size(), v.dtype, end="\t")
        pass

    import viz
    import numpy as np
    import matplotlib.pyplot as plt
    # viz.plot_2d(sample['pose2d'])
    viz.plot_mayavi(sample['pose3d'], sample['pose3d'])
    plt.imshow(np.transpose(
        sample['image'].numpy(), (1, 2, 0)).astype(np.float32))

    plt.show()
    # print(sample['pose2d'], '\n\n\n')
    # print(sample['pose3d'])

    del dataset
    del sample
    gc.collect()


if __name__ == "__main__":
    test_mpiinf()
