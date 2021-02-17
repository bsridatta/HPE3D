# %%
from torch import sub
from typing import List
import h5py
import glob
import os

from .data_utils import get_projection_params, project_to_2d


data_path: str = f'{os.getenv("HOME")}/lab/HPE_datasets/h36m/'

SUBJECTS: List[int] = [1, 5, 6, 7, 8, 9, 11]




# Convert all subjects from 3D to 2D
for subject in SUBJECTS:
    paths_3d: List[str] = glob.glob(
        data_path + f"S{subject}/MyPoses/3D_positions/*.h5")

    for path in paths_3d:
        h5 = h5py.File(path, 'r')
        poses_3d = h5["3D_positions"]

        # For each 3D h5 create 2D h5 from all 4 cameras
        for camera in range(1, 5):
            camera_params = get_projection_params(camera, subject)
            project_to_2d(poses_3d, camera_params)
# %%
