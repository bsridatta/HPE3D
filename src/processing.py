'''
Reference code/procedure - https://github.com/una-dinosauria/3d-pose-baseline
'''
import gc
import os
import math

import h5py
import numpy as np
import torch
from scipy.spatial import procrustes as proc


def zero_the_root(pose, root_idx):
    '''
    TODO UPDATE
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


def preprocess(data, joint_names, root_idx, normalize_pose=True, projection=True):
    pose2d = data['pose2d']
    pose3d = data['pose3d']
    assert pose2d.shape[1:] == (16,2)

    assert pose3d.shape[1:] == (16,3)
    # Scale 2D pose such that mean dist from head to root is 1/konwn_constant
    if projection:
        # calculate scale required to make 2D to 1/c unit
        c = 10

        # calculate the total distance between the head and the root 
        # 2D poses stil have 17 joints
        head2neck = np.linalg.norm(
            pose2d[:, joint_names.index('Head'), :] - pose2d[:, joint_names.index('Neck'), :], axis=1, keepdims=True)
        neck2torso = np.linalg.norm(
            pose2d[:, joint_names.index('Neck'), :] - pose2d[:, joint_names.index('Torso'), :], axis=1, keepdims=True)
        torso2root = np.linalg.norm(
            pose2d[:, joint_names.index('Torso'), :] - pose2d[:, joint_names.index('Pelvis'), :], axis=1, keepdims=True)
        dist = head2neck + neck2torso + torso2root

        # Google's poem scales using lower half 
        # head2neck = np.linalg.norm(
        #     pose2d[:,js.index('R_Hip'),:] - pose2d[:,js.index('R_Knee'),:], axis=1, keepdims=True)
        # neck2torso = np.linalg.norm(
        #     pose2d[:,js.index('R_Knee'),:] - pose2d[:,js.index('R_Ankle'),:], axis=1, keepdims=True)
        # dist = head2neck+neck2torso

        scale_2d = c*np.mean(dist)  # 1/c units
        pose2d = np.divide(pose2d.T, scale_2d.T).T

    # center the 2d and 3d pose at the root and remove the root
    pose2d = zero_the_root(pose2d, root_idx)
    pose3d = zero_the_root(pose3d, root_idx)

    if normalize_pose and not projection:
        # TODO UPDATE
        pose2d = normalize(pose2d)
        pose3d = normalize(pose3d)

    data['pose2d'] = pose2d
    data['pose3d'] = pose3d

    return data


def post_process(recon, target, scale=None, self_supervised=False, procrustes_enabled=False):
    '''
    DeNormalize Validation 3D poses alone - to calc the evaluation metric
    Add root at 0,0,0
    '''
    # TODO join root addition for supervised and self-supervised

    if not self_supervised:
        # de-normalize data to original coordinates
        recon = denormalize(recon)
        target = denormalize(target)

        # since the MPJPE is computed for 17 joints with roots aligned i.e zeroed
        recon = torch.cat(
            (torch.zeros(recon.shape[0], 1, recon.shape[2], device=recon.device), recon), dim=1)
        target = torch.cat(
            (torch.zeros(target.shape[0], 1, recon.shape[2], device=recon.device), target), dim=1)

    if procrustes_enabled:
        # algined recon with target
        t, r = target.cpu().numpy(), recon.cpu().numpy()

        aligned = []
        for t_, r_ in zip(t, r):
            # recon should be the second matrix
            _, mtx, _ = proc(t_, r_)
            mean = np.mean(t_, 0)
            std = np.linalg.norm(t_ - mean)
            r_ = (mtx*std) + mean
            aligned.append(r_)

        recon = torch.from_numpy(np.array(aligned)).float()

    if self_supervised:
        recon = torch.cat(
            (torch.tensor((0, 0, 0), device=recon.device, dtype=torch.float32).repeat(
                recon.shape[0], 1, 1),
                recon
             ), dim=1)
        # recon += torch.tensor((0, 0, 10), device=recon.device, dtype=torch.float32)

        target = torch.cat(
            (torch.tensor((0, 0, 0), device=recon.device, dtype=torch.float32).repeat(
                target.shape[0], 1, 1),
                target
             ), dim=1)
        # target += torch.tensor((0, 0, 10), device=recon.device, dtype=torch.float32)

    return recon, target


def create_rotation_matrices_3d(azimuths, elevations, rolls):
    '''
    https://github.com/google-research/google-research/tree/68c738421186ce85339bfee16bf3ca2ea3ec16e4/poem
    '''

    azi_cos = torch.cos(azimuths)
    azi_sin = torch.sin(azimuths)
    ele_cos = torch.cos(elevations)
    ele_sin = torch.sin(elevations)
    rol_cos = torch.cos(rolls)
    rol_sin = torch.sin(rolls)
    rotations_00 = azi_cos * ele_cos
    rotations_01 = azi_cos * ele_sin * rol_sin - azi_sin * rol_cos
    rotations_02 = azi_cos * ele_sin * rol_cos + azi_sin * rol_sin
    rotations_10 = azi_sin * ele_cos
    rotations_11 = azi_sin * ele_sin * rol_sin + azi_cos * rol_cos
    rotations_12 = azi_sin * ele_sin * rol_cos - azi_cos * rol_sin
    rotations_20 = -ele_sin
    rotations_21 = ele_cos * rol_sin
    rotations_22 = ele_cos * rol_cos
    rotations_0 = torch.stack(
        [rotations_00, rotations_10, rotations_20], axis=-1)
    rotations_1 = torch.stack(
        [rotations_01, rotations_11, rotations_21], axis=-1)
    rotations_2 = torch.stack(
        [rotations_02, rotations_12, rotations_22], axis=-1)

    return torch.stack([rotations_0, rotations_1, rotations_2], axis=-1)


def random_rotate(pose_3d,
                  roll_range=(0, 0),
                  azimuth_range=(0, 0),
                  elevation_range=(-math.pi, math.pi)
                  ):
    #   roll_range=(-math.pi / 9.0,
    #               math.pi / 9.0),
    """roll_range is elevation as x and y arent swapped"""
    azimuths = torch.rand(pose_3d.shape[:-2]) * \
        (azimuth_range[0]-azimuth_range[1]) + azimuth_range[1]

    perpendicular_novel_view = False
    if perpendicular_novel_view:
        elevations = torch.ones(pose_3d.shape[:-2]) * math.pi / \
            2 * (2*(torch.randint(-1, 1, pose_3d.shape[:-2])+0.5))
    else:
        elevations = torch.rand(pose_3d.shape[:-2]) * \
            (elevation_range[0]-elevation_range[1]) + elevation_range[1]

    rolls = torch.rand(pose_3d.shape[:-2]) * \
        (roll_range[0]-roll_range[1]) + roll_range[1]

    rotation_matrices = create_rotation_matrices_3d(
        azimuths, elevations, rolls)
    rotation_matrices = rotation_matrices.to(device=pose_3d.device)

    pose_3d_rotated = torch.matmul(
        rotation_matrices, torch.transpose(pose_3d, 1, 2))
    pose_3d_rotated = torch.transpose(pose_3d_rotated, 1, 2)

    return pose_3d_rotated


def project_3d_to_2d(pose_3d):

    pose_3d_z = (torch.clamp(pose_3d[Ellipsis, -1:], min=1e-12))
    pose_2d_reprojection = pose_3d[Ellipsis, :-1]/pose_3d_z

    return pose_2d_reprojection

################################################################################
# TODO not used - for supervised and images


# TODO think of another way to store norm_stats
path = f"{os.path.dirname(os.path.abspath(__file__))}/data/norm_stats.h5"
if os.path.exists(path):
    f = h5py.File(path, 'r')

    NORM_STATS = {}
    for key in f.keys():
        NORM_STATS[key] = np.array(f.get(key))[:]
    f.close()


def normalize(pose):
    '''
    TODO UPDATE
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
    """ 
    TODO UPDATE
    De-Standardize and then 
    Denormalize poses for evaluation of actual pose.  

    Args:
        pose (numpy): 2d/3d pose [n, 16, 2/3]

    Returns:
       numpy : denormalized pose [n, 16, 2/3]
    """

    pose *= torch.tensor(NORM_STATS[f"std{pose.shape[2]}d"],
                         device=pose.device)
    pose += torch.tensor(NORM_STATS[f"mean{pose.shape[2]}d"],
                         device=pose.device)

    return pose


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
