"""
Reference code/procedure - https://github.com/una-dinosauria/3d-pose-baseline
"""
import gc
import math
import os
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import torch
from scipy.spatial import procrustes as proc


def zero_the_root(poses: np.ndarray, root_idx: int) -> np.ndarray:
    """move pose such that root/pelvis is at origin

    Args:
        poses (np.ndarray): n poses
        root_idx (int): index of root(pelvis)

    Returns:
        poses (np.ndarray) -- poses with root shifted to origin,
                        w/o root as always 0
    """
    # center at root
    for i in range(poses.shape[0]):
        poses[i, :, :] = poses[i, :, :] - poses[i, root_idx, :]

    # remove root
    poses = np.delete(poses, root_idx, 1)  # axis -> [n, j, 2(x,y)]

    return poses

def scale_3d(poses):
    # TODO scale such that each poses upper half is of scale 1

    # enforce unit recon if above root is scaled to 1
    # tanh gives 0 to 1-  lower is 1 then upper is 0.8 we need upper 1    
    return poses*1.3

def preprocess(
    poses: np.ndarray,
    joint_names: List[str],
    root_idx: int,
    normalize_pose: bool = True,
    is_ss: bool = True,
    project_dist: float = 10,
) -> np.ndarray:
    """Normalize 2D, 3D for supervised or scale 2D for self supervised. Zero poses at root and remove roots.

    Args:
        poses (np.ndarray): 2D/3D poses 
        joint_names (List[str]): joint names in the order of the points in the dataset - taken care in dataset creation code
        root_idx (int): index of root
        normalize_pose (bool, optional): If supervised. Defaults to True.
        is_ss (bool, optional): True if self supervised training. Defaults to True.
        project_dist (float, optional): Distance of the image plane from camera. A unit 3D pose projected to 2D will be the inverse of this value. Defaults to 10.

    Returns:
        Dict: The dictionary with process pose values
    """

    assert poses.shape[-1] in [2, 3]

    # Scale 2D pose such that mean dist from head to root is 1/konwn_constant
    if is_ss and poses.shape[-1] == 2:
        # calculate scale required to make 2D to 1/c unit
        c = project_dist

        # calculate the total distance between the head and the root
        # 2D poses stil have 17 joints
        head2neck = np.linalg.norm(
            poses[:, joint_names.index("Head"), :]
            - poses[:, joint_names.index("Neck"), :],
            axis=1,
            keepdims=True,
        )
        neck2torso = np.linalg.norm(
            poses[:, joint_names.index("Neck"), :]
            - poses[:, joint_names.index("Torso"), :],
            axis=1,
            keepdims=True,
        )
        torso2root = np.linalg.norm(
            poses[:, joint_names.index("Torso"), :]
            - poses[:, joint_names.index("Pelvis"), :],
            axis=1,
            keepdims=True,
        )
        dist = head2neck + neck2torso + torso2root

        # Google's poem scales using lower half
        # head2neck = np.linalg.norm(
        #     poses[:,js.index('R_Hip'),:] - poses[:,js.index('R_Knee'),:], axis=1, keepdims=True)
        # neck2torso = np.linalg.norm(
        #     poses[:,js.index('R_Knee'),:] - poses[:,js.index('R_Ankle'),:], axis=1, keepdims=True)
        # dist = head2neck+neck2torso

        scale_2d = c * np.mean(dist)  # 1/c units
        poses = np.divide(poses.T, scale_2d.T).T  # type: ignore

    # center the 2d/3d pose at the root and remove the root
    poses = zero_the_root(poses, root_idx)

    # do not normalize if ss i.e using gans as normalization is done above by scaling 2d
    if normalize_pose and not is_ss:
        poses = normalize(poses)

    assert poses.shape[-2] == 15
    return poses


def post_process(
    recon, target, is_ss=True, procrustes=True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Denormalize or rigid body transformation of 3D poses for calculating eval metric. Adds root at 0,0,0 to get the metric for 17 joints.

    Args:
        recon (torch.Tensor): predicted 3D pose
        target (torch.Tensor): ground truth 3D pose
        is_ss (bool, optional): true if self supervised. Defaults to True.
        procrustes (bool, optional): if true, do procrustes alignemnt of recon to target. Defaults to True.

    Returns:
        Tuple: modified recon and target
    """

    # TODO remove adding root for 16j metric
    assert recon.shape == target.shape

    if not is_ss:
        # de-normalize data to original coordinates
        recon = denormalize(recon)
        target = denormalize(target)

    if procrustes:
        # Speed up procrustes alignment with CPU!
        t, r = target.cpu().numpy(), recon.cpu().numpy()

        aligned = []
        for t_, r_ in zip(t, r):
            # recon should be the second matrix
            _, mtx, _ = proc(t_, r_)
            mean = np.mean(t_, 0)
            std = np.linalg.norm(t_ - mean)
            r_ = (mtx * std) + mean
            aligned.append(r_)

        recon = torch.from_numpy(np.array(aligned)).float().to(recon.device)

    # since the MPJPE is computed for 17 joints with roots aligned i.e zeroed
    zeros = torch.tensor((0, 0, 0), device=recon.device, dtype=torch.float32).repeat(
        recon.shape[0], 1, 1
    ).to(recon.device)

    recon = torch.cat((zeros, recon), dim=1)
    target = torch.cat((zeros, target), dim=1)

    return recon, target


def create_rotation_matrices_3d(azimuths, elevations, rolls):
    """
    https://github.com/google-research/google-research/tree/68c738421186ce85339bfee16bf3ca2ea3ec16e4/poem
    """

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
    rotations_0 = torch.stack([rotations_00, rotations_10, rotations_20], axis=-1)
    rotations_1 = torch.stack([rotations_01, rotations_11, rotations_21], axis=-1)
    rotations_2 = torch.stack([rotations_02, rotations_12, rotations_22], axis=-1)

    return torch.stack([rotations_0, rotations_1, rotations_2], axis=-1)


def random_rotate(
    pose_3d,
    roll_range=(0, 0),
    azimuth_range=(0, 0),
    elevation_range=(-math.pi, math.pi),
):
    #   roll_range=(-math.pi / 9.0,
    #               math.pi / 9.0),
    """roll_range is elevation as x and y arent swapped"""
    azimuths = (
        torch.rand(pose_3d.shape[:-2]) * (azimuth_range[0] - azimuth_range[1])
        + azimuth_range[1]
    )

    perpendicular_novel_view = False
    if perpendicular_novel_view:
        elevations = (
            torch.ones(pose_3d.shape[:-2])
            * math.pi
            / 2
            * (2 * (torch.randint(-1, 1, pose_3d.shape[:-2]) + 0.5))
        )
    else:
        elevations = (
            torch.rand(pose_3d.shape[:-2]) * (elevation_range[0] - elevation_range[1])
            + elevation_range[1]
        )

    rolls = (
        torch.rand(pose_3d.shape[:-2]) * (roll_range[0] - roll_range[1]) + roll_range[1]
    )

    rotation_matrices = create_rotation_matrices_3d(azimuths, elevations, rolls)
    rotation_matrices = rotation_matrices.to(device=pose_3d.device)

    pose_3d_rotated = torch.matmul(rotation_matrices, torch.transpose(pose_3d, 1, 2))
    pose_3d_rotated = torch.transpose(pose_3d_rotated, 1, 2)

    return pose_3d_rotated


def translate_and_project(pose3d, project_dist):
    T = torch.tensor((0, 0, project_dist)).to(pose3d).type_as(pose3d)
    pose3d = pose3d + T
    pose3d_z = torch.clamp(pose3d[Ellipsis, -1:], min=1e-12)
    pose2d_reprojection = pose3d[Ellipsis, :-1] / pose3d_z

    return pose2d_reprojection


################################################################################
# TODO not used - for supervised and images


def normalize(pose):
    """
    TODO UPDATE
    using the mean and std of h3.6m trainig poses after zeroing the root and standarding
    refer -- src/data_preparation/h36m_annotations.py

    Arguments:
        pose (numpy) -- array of 2/3d poses with 16 joints w/o root [n, 16, 2/3]
    Returns:
        pose (numpy) -- array of normalized pose
    """
    mean = NORM_STATS[f"mean{pose.shape[2]}d"]
    pose = (pose - NORM_STATS[f"mean{pose.shape[2]}d"]) / NORM_STATS[
        f"std{pose.shape[2]}d"
    ]

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

    pose *= torch.tensor(NORM_STATS[f"std{pose.shape[2]}d"], device=pose.device)
    pose += torch.tensor(NORM_STATS[f"mean{pose.shape[2]}d"], device=pose.device)

    return pose
