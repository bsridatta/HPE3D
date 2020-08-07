import gc
import os

import albumentations
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.processing import preprocess


class H36M(Dataset):

    def __init__(self, subjects, annotation_file, image_path, no_images=False, device='cpu', annotation_path=None, train=False, projection=True):
        """

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

        self.no_images = no_images  # if only lifting 2D-3D
        self.device = device
        self.train = train
        self.image_path = image_path  # load image directly in __getitem__

        # Data Specific Information
        self.skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                         (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
        self.joint_names = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso',
                            'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')
        self.action_names = ["Directions", "Discussion", "Eating", "Greeting", "Phoning", "Photo", "Posing", "Purchases",
                             "Sitting", "SittingDown", "Smoking", "Waiting", "WalkDog", "Walking", "WalkTogether"]
        self.root_idx = self.joint_names.index('Pelvis')

        # fliped indices for 16 joints
        self.flipped_indices = [3, 4, 5, 0, 1, 2,
                                6, 7, 8, 9, 13, 14, 15, 10, 11, 12]

        ignore_data = ["pose3d_global"]

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

        # get keys to avoid query them for every __getitem__
        self.annotation_keys = self.annotations.keys()
        
        # further process to make the data learnable - zero 3dpose and norm poses
        print(f'[INFO]: processing subjects: {subjects}')
        self.annotations = preprocess(
            self.annotations, self.root_idx, normalize_pose=True, projection=projection)

        # covert data to tensor after preprocessing them as numpys (messy with tensors)
        for key in self.annotation_keys:
            self.annotations[key] = torch.tensor(
                self.annotations[key], dtype=torch.float32)

        # Image normalization
        self.augmentations = albumentations.Compose([
            albumentations.Normalize(always_apply=True)
        ])

        self.dataset_len = len(self.annotations['idx'])

        # clear the HDF5 datasets
        annotations_h5.close()
        del annotations_h5
        gc.collect()

    def __len__(self):
        # idx - index of the image files
        return self.dataset_len

    def __getitem__(self, idx):
        sample = {}
        for key in self.annotation_keys:
            sample[key] = self.annotations[key][idx]

        if not self.no_images:
            image = self.get_image_tensor(sample)
            sample['image'] = image

        if self.train and np.random.random() < 0.2:
            sample = self.flip(sample)

        return sample

    def get_image_tensor(self, sample):
        image_dir = 's_%02d_act_%02d_subact_%02d_ca_%02d'\
            % (sample['subject'], sample['action'],
               sample['subaction'], sample['camera'])
        image_ = Image.open(self.image_path+image_dir+'/' +
                            image_dir+"_"+("%06d" % (sample['idx']))+".jpg")

        image = np.array(image_)
        image = self.augmentations(image=image)['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float32)

        # image_.close()
        # del image_
        return image

    def flip(self, sample):
        sample['pose2d'] = sample['pose2d'][self.flipped_indices]
        sample['pose3d'] = sample['pose3d'][self.flipped_indices]
        sample['pose2d'][:, 0] *= -1
        sample['pose3d'][:, 0] *= -1
        # TODO add image flipping
        return sample


def test_h36m():
    '''
    Can be used to get norm stats for all subjects
    '''

    annotation_file = f'h36m17'
    image_path = f"{os.getenv('HOME')}/lab/HPE_datasets/h36m/"

    dataset = H36M([1, 5, 6, 7, 8],
                   annotation_file, image_path, train=True, no_images=False)

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
    test_h36m()
