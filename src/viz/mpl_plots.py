import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from torchvision import transforms
from src.processing import project_3d_to_2d

SKELETON_COLORS = ['b', 'b', 'b', 'b', 'orange', 'orange', 'orange',
                   'b', 'b', 'b', 'b', 'b', 'b', 'orange', 'orange', 'orange', 'orange']
SKELETON = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
            (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
LABELS = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
          'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')


def plot_2d(pose, mode="show", color=None, labels=False):
    """Base function for 2D pose plotting, choose from 'show', 'axis'

    Args:
        pose (array): for 16 joints its numpy array (for adding root joint)
                      else tensor also works
        mode (str): choice of what to do
            Image: plot -> png -> image tensor, useful for latent space viz 
            Axis: return only the matplotlib axis to plot via caller method.
            Show: Just show the plot 
    Returns:
        Tensor/MatplotlibAxis: Depends on the mode
    """
    fig = plt.figure(1)
    ax = fig.gca()

    if pose.shape[0] == 16:
        pose = np.concatenate((np.zeros((1, 2)), pose), axis=0)

    x = pose[:, 0]
    y = -1*pose[:, 1]

    ax.scatter(x, y, alpha=0.6, s=2)

    if color:
        colors = [color]*len(SKELETON_COLORS)
    else:
        colors = SKELETON_COLORS

    for link, color in zip(SKELETON, colors):
        ax.plot(x[([link[0], link[1]])],
                y[([link[0], link[1]])],
                c=color, alpha=0.6, lw=3)

    if labels:
        for i, j, l in zip(x, y, LABELS):
            ax.text(i, j, s=l, size=8, zorder=1, color='k')

    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    if mode == 'axis':
        return ax
    elif mode == 'show':
        plt.show()
    elif mode == "image":
        DPI = fig.get_dpi()
        fig.set_size_inches(305.0/float(DPI), 305.0/float(DPI))
        fig.savefig(f"{os.getenv('HOME')}/lab/HPE3D/src/results/x.png")
        fig.clf()
        image = Image.open(f"{os.getenv('HOME')}/lab/HPE3D/src/results/x.png")
        image = image.convert('RGB')
        image = transforms.ToTensor()(image).unsqueeze_(0)
        return image
    else:
        raise ValueError("Please choose from 'image', 'show', 'axis' only")


def plot_3d(pose, mode="show", color=None, floor=False, axis3don=True, labels=False):
    """Base function for 3D pose plotting, choose from 'show', 'image', 'axis'

    Args:
        pose (array): for 16 joints its numpy array (for adding root joint)
                      else tensor also works
        mode (str): choice of what to do
            Image: plot -> png -> image tensor, useful for latent space viz 
            Axis: return only the matplotlib axis to plot via caller method.
            Show: Just show the plot 
    Returns:
        Tensor/MatplotlibAxis: Depends on the mode

    """
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax._axis3don = axis3don

    if pose.shape[0] == 16:
        pose = np.concatenate((np.zeros((1, 3)), pose), axis=0)

    x = pose[:, 0]
    y = -1*pose[:, 2]
    z = -1*pose[:, 1]

    ax.scatter(x, y, z, alpha=0.6, s=0.1)

    if color:
        colors = [color]*len(SKELETON_COLORS)
    else:
        colors = SKELETON_COLORS

    for link, color_ in zip(SKELETON, colors):
        ax.plot(x[([link[0], link[1]])],
                y[([link[0], link[1]])],
                z[([link[0], link[1]])],
                c=color_, alpha=0.6, lw=3)

    fix_3D_aspect(ax, x, y, z)

    if labels:
        # Show coordinate values
        for i, j, k, l in zip(x, y, z, LABELS):
            ax.text(i, j, k, s=f'{l}', size=7, zorder=1, color='k')

    # Plot floor
    if floor:
        # xx, yy = np.meshgrid([-max(x), max(x)], [-max(y), max(y)])
        xx, yy = np.meshgrid([x[3], x[6]], [y[3], y[5]])
        zz = np.ones((len(xx), len(yy))) * min(z)*1.01  # padding
        ax.plot_surface(xx, yy, zz, cmap='gray',
                        linewidth=0, alpha=0.2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if mode == "axis":
        return ax
    elif mode == "show":
        plt.show()
    elif mode == "image":
        DPI = fig.get_dpi()
        fig.set_size_inches(305.0/float(DPI), 305.0/float(DPI))
        fig.savefig(f"{os.getenv('HOME')}/lab/HPE3D/src/results/x.png")
        fig.clf()
        image = Image.open(f"{os.getenv('HOME')}/lab/HPE3D/src/results/x.png")
        image = image.convert('RGB')
        image = transforms.ToTensor()(image).unsqueeze_(0)
        return image
    else:
        raise ValueError("Please choose from 'image', 'show', 'axis' only")


def fix_3D_aspect(ax, x, y, z):
    """From https://stackoverflow.com/a/13701747/6710388"""
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array(
        [x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -
                                1:2:2][0].flatten() + 0.5*(x.max()+x.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -
                                1:2:2][1].flatten() + 0.5*(y.max()+y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -
                                1:2:2][2].flatten() + 0.5*(z.max()+z.min())

    # Comment or uncomment the below lines to test the fake bounding box
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')


def plot_area(pose1, pose2):
    """Plot area between 2 poses, used for showing error in prediction

    Args:
        pose1 (array): predicted 
        pose2 (array): ground truth
    """
    x1 = pose1[:, 0]
    y1 = -1*pose1[:, 2]
    z1 = -1*pose1[:, 1]

    x2 = pose2[:, 0]
    y2 = -1*pose2[:, 2]
    z2 = -1*pose2[:, 1]

    vertices = []
    for link in SKELETON:
        area = [(x1[link[0]], y1[link[0]], z1[link[0]]),
                (x1[link[1]], y1[link[1]], z1[link[1]]),
                (x2[link[1]], y2[link[1]], z2[link[1]]),
                (x2[link[0]], y2[link[0]], z2[link[0]])
                ]
        vertices.append(area)

    fig = plt.figure(1)
    ax = fig.gca()

    ax.add_collection3d(Poly3DCollection(vertices, facecolors=[
                        'r', 'r'], alpha=0.2, zorder='max'))


def plot_data(pose2d=None, pose3d=None, image=None):
    """creates a combined figure with image 2d and 3d plots

    Args:
        pose2d (numpy array): 2d pose
        pose3d (numpy array): 3d pose
        image ([type], optional): image
    """
    fig = plt.figure()
    i = 1
    col = 0

    for x in [pose2d, pose3d, image]:
        if x is not None:
            col += 1

    if image is not None:
        ax = fig.add_subplot(100+col*10+i)
        i += 1
        if type(image) == str:
            image = Image.open(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image.convert("RGB"))

    if pose2d is not None:
        ax = fig.add_subplot(100+col*10+i)
        i += 1
        plot_2d(pose2d, mode='axis')

    if pose3d is not None:
        ax = fig.add_subplot(100+col*10+i, projection='3d')
        i += 1
        plot_3d(pose3d, mode="axis")

    plt.show()


def plot_errors(poses, targets, errors=None, grid=2):
    """Show difference between predictions and targets

    Arguments:
        poses {list} -- list of prediction
        targets {list} -- list of ground truths
        errors {list} -- list of avg pjpe for each prediction

    Keyword Arguments:
        grid {int} -- number of plots to show (default: {5})
    """
    fig = plt.figure(figsize=(15., 12.))
    plt.rcParams['savefig.dpi'] = 300

    rows = cols = grid

    i = 1
    for pose, target in zip(poses, targets):
        ax = fig.add_subplot(rows, cols, i, projection='3d')
        plot_3d(pose, mode="axis", color='b', floor=True, axis3don=False)
        plot_3d(target, mode="axis", color='grey', floor=True, axis3don=False)
        plot_area(pose, target)
        i += 1

    plt.show()


def print_pose(pose):
    """print pose with its joint name. for debugging

    Args:
        pose (numpy): 2D or 3D pose
    """
    joint_names = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso',
                   'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')

    if torch.is_tensor(pose):
        pose = pose.numpy()
    if len(pose) == 17:
        for x in range(len(pose)):
            print(f'{joint_names[x]:10} {pose[x]}')
    else:
        for x in range(len(pose)):
            print(f'{joint_names[x+1]:10} {pose[x]}')


def plot_projection(sample):
    fig = plt.figure()
    i = 1
    col = 3

    pose2d = sample['pose2d']
    # pose2d -= pose2d[0]
    pose3d = sample['pose3d']
    # pose3d -= pose3d[0]

    # if pose2d:
    ax = fig.add_subplot(100+col*10+i)
    i += 1
    plot_2d(pose2d, mode='axis')

    # if pose3d:
    ax = fig.add_subplot(100+col*10+i, projection='3d')
    i += 1
    plot_3d(pose3d, mode="axis")
    
    # if pose2d_proj:
    pose2d_proj = project_3d_to_2d(pose3d, cam_params=sample)

    ax = fig.add_subplot(100+col*10+i)
    i += 1
    plot_2d(pose2d_proj, mode='axis')
    plt.savefig("test.svg")
    plt.show()