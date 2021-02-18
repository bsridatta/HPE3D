# %%
from collections import defaultdict
from src.data_preparation.h36m_utils import action_to_id, camera_id_to_num
import os
import h5py
import numpy as np
import glob

DATA_PATH: str = f'{os.getenv("HOME")}/lab/HPE_datasets/h36m/'

TRAIN: bool = False
TYPE_2D = ["GT_2D", "SH", "SH_FT"][2]

if TRAIN:
    subject_list = [1, 5, 6, 7, 8]
    skip_frames = 5
    dataset_name = f"h36_train_{TYPE_2D.lower()}"

else:
    subject_list = [9, 11]
    skip_frames = 64
    dataset_name = f"h36_test_{TYPE_2D.lower()}"

dataset = defaultdict(list)
print(f"Loading {TYPE_2D} h5s from {DATA_PATH}")

for subject in subject_list:
    print(f"Preparing subject {subject} ...")

    # ../h36m/S[subject]/[GT_2D, SH, GT_3D]/[action]_[subaction].[camera_id].h5
    paths = glob.glob(DATA_PATH + f"S{subject}/{TYPE_2D}/*.h5")

    # all paths are same except for the parent directory name
    for path in paths:
        filename = path.split('/')[-1]
        action_subaction = filename.split('.')[0].split("_")
        action = action_subaction[0]
        subaction = 0 if len(action_subaction) == 1 else action_subaction[1]
        camera_id = filename.split('.')[1]

        if subject == 11 and action == 'Directions' and subaction == 0:
            continue  # corrupt recording

        pose2d_ = np.array(h5py.File(path, 'r')['poses'])
        dataset['pose2d'].extend(list(pose2d_)[::skip_frames])

        # get identical 3D data
        path_3d = path.replace(TYPE_2D, "GT_3D")
        pose3d_ = np.array(h5py.File(path_3d, 'r')['poses'])
        dataset['pose3d'].extend(list(pose3d_)[::skip_frames])

        # metadata - data imp. for analysis and reading images
        # can be used to id the source h5 file
        dataset['subject'].extend(
            pose2d_.shape[0]*[subject])
        dataset['action'].extend(
            pose2d_.shape[0]*[action_to_id(action)])
        dataset['subaction'].extend(
            pose2d_.shape[0]*[subaction])
        dataset['camera'].extend(
            pose2d_.shape[0]*[camera_id_to_num(int(camera_id))])

# keep track of order
dataset['idx'] = [*range(len(dataset['pose2d']))]

print("Total samples ", len(dataset['pose2d']))
print("Keys ", dataset.keys())

save_path = f"{os.getenv('HOME')}/lab/HPE3D/src/data/{dataset_name}.npy"
np.save(save_path, dataset)
print("Saved! ", save_path)
