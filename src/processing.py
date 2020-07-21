'''
Reference code - https://github.com/una-dinosauria/3d-pose-baseline
'''
import gc
import os

import h5py
import numpy as np
import torch

path = f"{os.path.dirname(os.path.abspath(__file__))}/data/norm_stats.h5"
if os.path.exists(path):
    f = h5py.File(path, 'r')

NORM_STATS = {}
for key in f.keys():
    NORM_STATS[key] = f[key][:]
f.close()

# path = f"{os.path.dirname(os.path.abspath(__file__))}/data/norm_stats_val.h5"
# if os.path.exists(path):
#     norm_stats_val = h5py.File(path, 'r')


ROOT_INDEX = 0  # Root at Pelvis index 0


def zero_the_root(pose, root_idx=ROOT_INDEX):
    '''
    center around root - pelvis

    Arguments:
        pose (numpy) -- array of poses
        root_idx (int) -- index of root(pelvis)

    Returns:
        pose (numpy) -- poses with root shifted to origin,
                        w/o root as always 0 
    '''
    # center at root
    for i in range(pose.shape[0]):
        pose[i, :, :] = pose[i, :, :] - pose[i, root_idx, :]

    # remove root
    pose = np.delete(pose, root_idx, 1)  # axis -> [n, j, x/y]

    return pose


def normalize(pose):
    '''
    using the mean and std of h3.6m trainig poses after zeroing the root and standarding
    refer -- src/data_preparation/h36m_annotations.py

    Arguments:
        pose (numpy) -- array of 2/3d poses with 16 joints w/o root [n, 16, 2/3]
    Returns:
        pose (numpy) -- array of normalized pose
    '''
    mean = NORM_STATS[f"mean{pose.shape[2]}d"]
    pose = (pose - NORM_STATS[f"mean{pose.shape[2]}d"]
            )/NORM_STATS[f"std{pose.shape[2]}d"]

    return pose


def denormalize(pose):
    """ De-Standardize and then 
    Denormalize poses for evaluation of actual pose.  

    Args:
        pose (numpy): 2d/3d pose [n, 16, 2/3]

    Returns:
       numpy : denormalized pose [n, 16, 2/3]
    """

    # pose *= torch.tensor(norm_stats_val[f"mean_dist{pose.shape[2]}d"],
    #                      device=pose.device).reshape((-1, 1, 1))

    pose *= torch.tensor(NORM_STATS[f"std{pose.shape[2]}d"],
                         device=pose.device)
    pose += torch.tensor(NORM_STATS[f"mean{pose.shape[2]}d"],
                         device=pose.device)

    return pose


def preprocess(annotations, root_idx=ROOT_INDEX, normalize_pose=True):
    '''
    Preprocessing steps on -
    pose3d - 3d pose in camera frame(data already coverted from world to camera)
    pose2d - 2d poses obtained by projection of above 3d pose using camera params

    Arguments:
        annotations (dic) -- dictionary of all data excluding raw images

    Returns:
        annotations (dic) -- with normalized 16 joint 2d and 3d poses
    '''
    # center the 2d and 3d pose at the root and remove the root
    pose2d = zero_the_root(annotations['pose2d'], root_idx)
    pose3d = zero_the_root(annotations['pose3d'], root_idx)

    if normalize_pose:
        # Standardize
        # mean_dist2d = np.mean(np.sqrt(
        #     np.sum(np.power(np.subtract(pose2d, np.zeros((1, 2))), 2), axis=2)), axis=1)
        # mean_dist3d = np.mean(np.sqrt(
        #     np.sum(np.power(np.subtract(pose3d, np.zeros((1, 3))), 2), axis=2)), axis=1)
        # pose2d = pose2d/mean_dist2d.reshape(-1, 1, 1)
        # pose3d = pose3d/mean_dist3d.reshape(-1, 1, 1)

        # # Normalize
        pose2d = normalize(pose2d)
        pose3d = normalize(pose3d)

    annotations['pose2d'] = pose2d
    annotations['pose3d'] = pose3d

    return annotations


def post_process(config, recon, target):
    '''
    DeNormalize Validation Data
    Add root at 0,0,0
    '''

    # de-normalize data to original coordinates
    recon = denormalize(recon)
    target = denormalize(target)

    # # # de-standardize
    # recon = recon*torch.tensor(NORM_STATS[f"max3d"],
    #                      device=recon.device)
    # target = target*torch.tensor(NORM_STATS[f"max3d"],
    #                      device=target.device)

    # since the MPJPE is computed for 17 joints with roots aligned i.e zeroed
    # Not very fair, but the average is with 17 in the denom after adding root joiny at zero!
    recon = torch.cat(
        (torch.zeros(recon.shape[0], 1, recon.shape[2], device=config.device), recon), dim=1)
    target = torch.cat(
        (torch.zeros(target.shape[0], 1, recon.shape[2], device=config.device), target), dim=1)

    return recon, target


def normalize_image(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0):
    """Code from Albumentations

    Arguments:
        img {Tensor} -- Image sample

    Keyword Arguments:
        mean {tuple} -- Mean of ImageNet used in albumentations (default: {(0.485, 0.456, 0.406)})
        std {tuple} -- Std of ImageNet (default: {(0.229, 0.224, 0.225)})
        max_pixel_value {float} -- Max value before normalization (default: {255.0})

    Returns:
        Tensor -- Normalized image
    """
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator

    return img


def project_3d_to_2d(pose, R, T, f, c,)

# if __name__ == "__main__":
#     for x in NORM_STATS.keys():
#         print(x, '\n', NORM_STATS[x])
