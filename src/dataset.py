import gc
import os

import albumentations
import h5py
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset

from processing import preprocess

# from PIL import Image


class H36M(Dataset):

    def __init__(self, subjects, annotation_file, image_path, no_images=False, device='cpu', annotation_path=None, train=False):
        """[summary]

        Arguments:
            subjects {list} -- IDs of subjects to include in this dataset
            annotation_file {str} -- file name (debug_h36m_17, h36m17 etc)
            image_path {str} -- path to image data folder

        Keyword Arguments:
            no_images {bool} -- To exclude images from samples (default: {False})
            device {str} -- (default: {'cpu'})
            annotation_path {str} -- path to the annotation_file
            train {bool} -- triggers data augmentation during training (default: {False})
        """

        self.no_images = no_images  # incase of only lifting 2D-3D
        self.device = device
        self.train = train

        # Data Specific Information
        self.skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                         (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        self.joint_names = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso',
                            'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.action_names = ["Directions", "Discussion", "Eating", "Greeting", "Phoning", "Photo", "Posing", "Purchases",
                             "Sitting", "SittingDown", "Smoking", "Waiting", "WalkDog", "Walking", "WalkTogether"]
        # self.flip_pairs = ((1, 4), (2, 5), (3, 6),
        #                    (14, 11), (15, 12), (16, 13))
        self.flipped_indices = [0, 4, 5, 6, 1, 2, 3,
                                7, 8, 9, 10, 14, 15, 16, 11, 12, 13]

        self.root_idx = self.joint_names.index('Pelvis')

        ignore_data = ["pose3d_global", "bbox",
                       "cam_f", "cam_c", "cam_R", "cam_T"]

        # get labels and metadata including camera parameters
        subj_name = "".join(str(sub) for sub in subjects)
        if annotation_path:
            annotations_h5 = h5py.File(
                f'{annotation_path}/{annotation_file}_{subj_name}.h5', 'r')
        else:
            annotations_h5 = h5py.File(
                f'{os.path.dirname(os.path.abspath(__file__))}/data/{annotation_file}_{subj_name}.h5', 'r')

        # store only the subjects of interest
        self.annotations = {}
        for key in annotations_h5.keys():
            if key not in ignore_data:
                self.annotations[key] = annotations_h5[key][:]

        # further process to make the data learnable - zero 3dpose and norm poses
        print(f'[INFO]: processing subjects: {subjects}')
        self.annotations, norm_stats = preprocess(
            self.annotations, self.root_idx)

        # get keys to avoid query them for every __getitem__
        self.annotation_keys = self.annotations.keys()

        # covert data to tensor after preprocessing with numpy (hard with tensors)
        for key in self.annotation_keys:
            self.annotations[key] = torch.tensor(
                self.annotations[key], dtype=torch.float32)

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

        # load image directly in __getitem__
        self.image_path = image_path

        self.augmentations = albumentations.Compose([
            albumentations.Normalize(always_apply=True)
        ])

    def __len__(self):
        # idx - index of the image files
        return len(self.annotations['idx'])

    def __getitem__(self, idx):

        # Get all data for a sample
        sample = {}
        for key in self.annotation_keys:
            sample[key] = self.annotations[key][idx]
        if not self.no_images:
            image = self.get_image_tensor(sample)
            sample['image'] = image

        # Augmentation - Flip
        if self.train and np.random.random() < 0.5:
            sample = self.flip(self, sample)

        return sample

    def get_image_tensor(self, sample):
        image_dir = 's_%02d_act_%02d_subact_%02d_ca_%02d'\
            % (sample['subject'], sample['action'],
               sample['subaction'], sample['camera'])

        image = joblib.load(self.image_path+image_dir+'/' +
                            image_dir+"_"+("%06d" % (sample['idx']))+".pkl")

        image = self.augmentations(image=image)['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float32)

        # *Note* - tranforms.ToTensor converts HWC to CHW so no need NOT to do explicitly
        # But if you do torch.tensor() you have to do it manually

        return image

    def flip(self, sample):
        pose2d_flip = sample['pose2d'].clone()
        pose3d_flip = sample['pose3d'].clone()

        for idx, x in self.flipped_indices:
            pose2d_flip[idx] = sample['pose2d'][x]
            pose3d_flip[idx] = sample['pose3d'][x]

        sample['pose2d'] = pose2d_flip
        sample['pose3d'] = pose3d_flip

        del pose2d_flip, pose3d_flip
        return sample

'''
Can be used to get norm stats for all subjects
'''


def test_h36m():
    annotation_file = f'debug_h36m17'
    image_path = f"/home/datta/lab/HPE_datasets/h36m_pickles/"

    dataset = H36M([1, 5, 6, 7, 8, 9, 11],
                   annotation_file, image_path, train=True)

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
