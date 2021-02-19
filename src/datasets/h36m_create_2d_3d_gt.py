from typing import List
import h5py
import glob
import os
import shutil
import numpy as np
from src.datasets.h36m_utils import *

DATA_PATH: str = f"{os.getenv('HOME')}/lab/HPE_datasets/h36m/"
SUBJECTS: List[int] = [1, 5, 6, 7, 8, 9, 11]
NUM_CAMS: int = 4


"""
Save all poses as ../h36m/S[subject]/[GT_2D, SH, GT_3D]/[action]_[subaction].[camera_id].h5
subaction is the suffix for actions 1,2,3 or "" (replaced as 0) 
"""

print("Using h5s from ", DATA_PATH)

# Convert all subjects from 3D to 2D
for subject in SUBJECTS:
    print(f"Creating subject {subject} ...")
    paths_3d: List[str] = glob.glob(
        DATA_PATH + f"S{subject}/MyPoses/3D_positions/*.h5")

    # create GT_2D and GT_3D folders in each subject
    save_dir = DATA_PATH+f"S{subject}/GT_2D/"
    if os.path.exists(save_dir):
        print("2D dir already exists, overwriting")
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    save_dir = save_dir.replace("GT_2D", "GT_3D")
    if os.path.exists(save_dir):
        print("3D dir already exists, overwriting")
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    for path in paths_3d:
        h5 = h5py.File(path, 'r')
        poses_3d = np.array(h5['3D_positions'])
        poses_3d = poses_3d.reshape(32, 3, -1).transpose(2, 0, 1)
        poses_3d /= 1000  # to meters
        poses_3d = poses_3d.astype('float32')
        poses_3d = remove_joints_from_3d(poses_3d)

        # For each 3D h5 create 2D h5 from all 4 cameras
        for camera in range(1, NUM_CAMS+1):
            # Transform world coordinates to camera coordinates
            R = get_extrinsic(subject, camera, param='orientation')
            t = get_extrinsic(subject, camera, param='translation')
            poses_3d_cam = world_to_camera(poses_3d, R, t)

            # Project to image plane
            projection_params = get_projection_params(camera)
            poses_2d = wrap(project_to_2d, poses_3d_cam,
                            projection_params, unsqueeze=True)

            # Transform to pixel/image coordinates
            res_w = get_intrinsic(camera, param='res_w')
            res_h = get_intrinsic(camera, param='res_h')
            poses_2d_pixel_space = image_coordinates(
                poses_2d, w=res_w, h=res_h)

            # save 2D GTs in separate folder
            path_2d = path.replace("MyPoses/3D_positions", "GT_2D")
            # h5s consistent with SH filenames ../Direction 1.h5 -> ../Directions_1.h5
            path_2d = path_2d.replace(" ", '_')
            # append camera id ../Direction_1.h5 -> ../Directions_1.58860488.h5
            path_2d = path_2d.replace('.h5', f".{camera_num_to_id(camera)}.h5")

            # Saving as ../S[subject]/[GT_2D, SH, GT_3D]/[action]_[subaction].[camera_id].h5
            h5 = h5py.File(path_2d, 'w')
            h5['poses'] = poses_2d_pixel_space
            h5.close()
            print(f"saved! {path_2d}")

            # save 3D poses in camera coordinates instead of world
            path_3d = path_2d.replace("GT_2D", "GT_3D")
            h5 = h5py.File(path_3d, 'w')
            h5['poses'] = poses_3d_cam
            h5.close()
            print(f"saved! {path_3d}")
