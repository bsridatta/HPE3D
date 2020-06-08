'''
Reference code - https://github.com/una-dinosauria/3d-pose-baseline
'''
import gc
import os

import h5py
import numpy as np
import torch

def zero_the_root(pose3d, root_idx):
    '''
    center around root - pelvis

    Arguments:
        pose3d (array) -- array of 3d poses of shape [17,3]
        root_idx (int) -- index of root(pelvis)

    Returns:
        pose3d (array) -- poses with root shifted to 0,0,0. 
                        w/o root as always 0 
    '''
    # center at root
    for i in range(pose3d.shape[0]):
        pose3d[i, :, :] = pose3d[i, :, :] - pose3d[i, root_idx, :]

    # remove root
    pose3d = np.delete(pose3d, root_idx, 1)  # axis [n, j, x/y]

    return pose3d


def normalize(pose):
    '''
    with mean, std with pose.shape ie for (x,y,z) of each joint

    Arguments:
        pose3d (array) -- array of 2/3d poses with 16 joints w/o root [n, 16, 2/3]

    Returns:
        pose_norm (array) -- array of normalized pose
        mean (array) -- mean of poses (mean pose) [16, 2/3]
        std (array) -- std of poses [16, 2/3]
    '''
    mean = np.mean(pose, axis=0)
    std = np.std(pose, axis=0)
    pose_norm = (pose - mean)/std

    return pose_norm, mean, std


def denormalize(pose, mean, std):
    '''
    Denormalize poses for evaluation of actual pose. Could also be obtained by just multiplying std
    '''
    pose *= std
    pose += mean

    return pose


def preprocess(annotations, root_idx):
    '''
    Preprocessing steps on -
    pose3d - 3d pose in camera frame(data already coverted from world to camera)
    pose2d - 2d poses obtained by projection of above 3d pose using camera params

    Arguments:
        annotations (dic) -- dictionary of all data excluding raw images

    Returns:
        annotations (dic) -- with normalized 16 joint 2d and 3d poses
        norm_stats (dic) -- mean and std of 2d, 3d poses to use for de-norm
    '''
    norm_stats = {}  # store mean and std of poses
    # center the 3d pose at the root and remove the root
    pose3d_zeroed = zero_the_root(annotations['pose3d'], root_idx)

    # remove root joint in 2d pose
    pose2d_16_joints = np.delete(
        annotations['pose2d'], root_idx, 1)  # axis [n, j, x/y/z]

    # normalize 2d and 3d poses
    pose2d_norm, norm_stats['mean2d'], norm_stats['std2d'] = normalize(
        pose2d_16_joints)

    pose3d_norm, norm_stats['mean3d'], norm_stats['std3d'] = normalize(
        pose3d_zeroed)

    annotations['pose2d'] = pose2d_norm
    annotations['pose3d'] = pose3d_norm

    return annotations, norm_stats


def post_process(config, recon, target):
    '''
    Normalize Validation Data
    Add root at 0,0
    '''
    ann_file_name = config.annotation_file.split('/')[-1].split('.')[0]
    norm_stats = h5py.File(
        f"{os.path.dirname(os.path.abspath(__file__))}/data/norm_stats_{ann_file_name}_911.h5", 'r')

    # de-normalize data to original positions
    recon = denormalize(
        recon,
        torch.tensor(norm_stats['mean3d'], device=config.device),
        torch.tensor(norm_stats['std3d'], device=config.device))
    target = denormalize(
        target,
        torch.tensor(norm_stats['mean3d'], device=config.device),
        torch.tensor(norm_stats['std3d'], device=config.device))

    # since the MPJPE is computed for 17 joints with roots aligned i.e zeroed
    # Not very fair, but the average is with 17 in the denom!
    recon = torch.cat(
        (torch.zeros(recon.shape[0], 1, 3, device=config.device), recon), dim=1)
    target = torch.cat(
        (torch.zeros(target.shape[0], 1, 3, device=config.device), target), dim=1)

    norm_stats.close()
    del norm_stats
    gc.collect()

    return recon, target
