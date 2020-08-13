'''
Reference code - https://github.com/una-dinosauria/3d-pose-baseline
'''
import gc
import os
import math

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

    pose *= torch.tensor(NORM_STATS[f"std{pose.shape[2]}d"],
                         device=pose.device)
    pose += torch.tensor(NORM_STATS[f"mean{pose.shape[2]}d"],
                         device=pose.device)

    return pose


def preprocess(annotations, root_idx=ROOT_INDEX, normalize_pose=True, projection=False):
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

    if normalize_pose and not projection:
        # Normalize
        pose2d = normalize(pose2d)
        pose3d = normalize(pose3d)

    elif projection:
        head = pose2d[:, 9, :]  # Note root joint removed
        root = np.zeros_like(head)
        dist = np.linalg.norm(head-root, axis=1, keepdims=True)
        scale = 10*dist
        pose2d = np.divide(pose2d.T, scale.T).T
        annotations['scale'] = scale

    annotations['pose2d'] = pose2d
    annotations['pose3d'] = pose3d

    return annotations


def post_process(config, recon, target, scale=None):
    '''
    DeNormalize Validation Data
    Add root at 0,0,0
    '''
    if not config.self_supervised:
        # de-normalize data to original coordinates
        recon = denormalize(recon)
        target = denormalize(target)

        # since the MPJPE is computed for 17 joints with roots aligned i.e zeroed
        # Not very fair, but the average is with 17 in the denom after adding root joiny at zero!
        recon = torch.cat(
            (torch.zeros(recon.shape[0], 1, recon.shape[2], device=config.device), recon), dim=1)
        target = torch.cat(
            (torch.zeros(target.shape[0], 1, recon.shape[2], device=config.device), target), dim=1)

    else:
        # de-scale
        recon = (recon.T*(scale*10).T).T
        recon = torch.cat(
            (torch.tensor((0, 0, 10), device=config.device, dtype=torch.float32).repeat(
                recon.shape[0], 1, 1),
                recon
             ), dim=1)

        target = torch.cat(
            (torch.tensor((0, 0, 10), device=config.device, dtype=torch.float32).repeat(
                target.shape[0], 1, 1),
                target
             ), dim=1)

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


def project_3d_to_2d(pose3d, cam_params, coordinates="camera"):
    """
    Project 3d pose from camera coordiantes to image frame
    using camera parameters including radial and tangential distortion

    Args:
        P: Nx3 points in camera coordinates
        cam_params (dic):
            R: 3x3 Camera rotation matrix
            T: 3x1 Camera translation parameters
            f: 2x1 Camera focal length accounting imprection in x, y
            c: 2x1 Camera center
            k: 3x1 Camera radial distortion coefficients
            p: 2x1 Camera tangential distortion coefficients

    Returns:
        Proj: Nx2 points in pixel space
        D: 1xN depth of each point in camera space
        radial: 1xN radial distortion per point
        tan: 1xN tangential distortion per point
        r2: 1xN squared radius of the projected points before distortion
    """
    # f = cam_params['cam_f'].view(-1, 1, 2)
    # c = cam_params['cam_c'].view(-1, 1, 2)

    if coordinates == 'world':  # rotate and translate
        R = cam_params('cam_R')
        T = cam_params('cam_T')
        # TODO maybe works only for one
        X = R.dot(torch.transpose(pose3d, 1, 2) - T)

    pose2d_proj = (pose3d/pose3d[:, :, 2][:, :, None].repeat(1, 1, 3))[:, :, :2]

    # f = 1
    # c = 0
    # pose2d_proj = f * pose2d_proj + c

    return pose2d_proj


def create_rotation_matrices_3d(azimuths, elevations, rolls):

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


def random_rotate_and_project_3d_to_2d(pose_3d,
                                       azimuth_range=(-math.pi, math.pi),
                                       elevation_range=(-math.pi / 6.0,
                                                        math.pi / 6.0),
                                       roll_range=(0.0, 0.0),
                                       default_camera=True,
                                       default_camera_z=10.0,
                                       random_rotate=True
                                       ):

    if random_rotate:
        azimuths = torch.rand(pose_3d.shape[:-2]) * \
            (azimuth_range[0]-azimuth_range[1]) + azimuth_range[1]
        elevations = torch.rand(pose_3d.shape[:-2]) * \
            (elevation_range[0]-elevation_range[1]) + elevation_range[1]
        rolls = torch.rand(pose_3d.shape[:-2]) * \
            (roll_range[0]-roll_range[1]) + roll_range[1]

        rotation_matrices = create_rotation_matrices_3d(azimuths, elevations, rolls)
        rotation_matrices = rotation_matrices.to(device=pose_3d.device)
        # TODO flip x, y
        pose_3d = torch.matmul(rotation_matrices, torch.transpose(pose_3d, 1, 2))
        pose_3d = torch.transpose(pose_3d, 1, 2)
   

    pose_3d_z = (torch.clamp(pose_3d[Ellipsis, -1:], min=1e-12))
    pose_2d_reprojection = pose_3d[Ellipsis, :-1]/pose_3d_z

    return pose_2d_reprojection
