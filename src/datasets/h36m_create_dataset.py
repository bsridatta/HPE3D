# %%
from collections import defaultdict
from src.datasets.h36m_utils import action_to_id, camera_id_to_num
import os
import h5py
import numpy as np
import glob


data_path: str = f'{os.getenv("HOME")}/lab/HPE_datasets/h36m/'
train: bool = False
type_2d = ["GT_2D", "SH", "SH_FT"][1]

if train:
    subject_list = [1, 5, 6, 7, 8]
    skip_frames = 5
    dataset_name = f"h36m_train_{type_2d.lower()}"

else:
    subject_list = [9, 11]
    skip_frames = 64
    dataset_name = f"h36m_test_{type_2d.lower()}"

dataset = defaultdict(list)
print(f"Loading {type_2d} h5s from {data_path}")

for subject in subject_list:
    print(f"Preparing subject {subject} ...")

    # ../h36m/S[subject]/[GT_2D, SH, GT_3D]/[action]_[subaction].[camera_id].h5
    paths = glob.glob(data_path + f"S{subject}/{type_2d}/*.h5")

    # all paths are same except for the parent directory name
    for path in paths:
        filename = path.split('/')[-1]
        action_subaction = filename.split('.')[0].split("_")
        action = action_subaction[0]
        subaction = 0 if len(action_subaction) == 1 else int(
            action_subaction[1])
        camera_id = filename.split('.')[1]

        if subject == 11 and action == 'Directions' and subaction == 0:
            continue  # corrupt recording

        pose2d_ = np.array(h5py.File(path, 'r')['poses'])
        pose2d_ = list(pose2d_)[::skip_frames]
        dataset['pose2d'].extend(pose2d_)

        # get identical 3D data
        # 3Ds corresponding to available 2Ds are only loaded
        path_3d = path.replace(type_2d, "GT_3D")
        pose3d_ = np.array(h5py.File(path_3d, 'r')['poses'])
        pose3d_ = list(pose3d_)[::skip_frames]
        dataset['pose3d'].extend(pose3d_)

        # metadata - data imp. for analysis and reading images
        # can be used to id the source h5 file
        dataset['subject'].extend(
            len(pose2d_)*[subject])
        dataset['action'].extend(
            len(pose2d_)*[action_to_id(action)])
        dataset['subaction'].extend(
            len(pose2d_)*[subaction])
        dataset['camera'].extend(
            len(pose2d_)*[camera_id_to_num(int(camera_id))])

# keep track of order
dataset['idx'] = [*range(len(dataset['pose2d']))]

print("Total samples ", len(dataset['pose2d']))
print("Keys ", dataset.keys())

save_path = f"{os.getenv('HOME')}/lab/HPE3D/src/data/{dataset_name}.h5"

h5 = h5py.File(save_path, 'w')
for key, value in dataset.items():
    print(key, len(value))
    h5[key] = value
h5.close()

print("Saved! ", save_path)
