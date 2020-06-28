'''
protocol #1 - using all 4 camera views in subjects S1, S5, S6, S7 and S8 for training and the 
same 4 camera views in subjects S9 and S11 for testing. 

protocol #2 - the predictions are post-processed via a rigid transformation
before comparing to the ground-truth

Eval metric - mean per joint positioning error (MPJPE)
Protocol-I computes the MPJPE directly whereas protocol-II first employs a rigid 
alignment between the poses. For a sequence the MPJPE’s are summed and 
divided by the number of frames
'''
########### Metadata #############
# Human 3.6
# actions = 15
# subaction = 2
# n_subject = 7
# subjects = [1 5 6 7 8 9 11]
# camera = [1 2 3 4]
# joints = 17
# action ()bbox (4,)cam_R (9,)cam_T (3,)cam_c (2,)cam_f (2,)camera ()
# idx ()pose2d (17, 2)pose3d (17, 3)pose3d_global (17, 3)subaction ()subject ()
# # 1,2 are standard training protocols for Human3.6m, the rest is for experiments
# protocols = {
#     1: {"train": [1, 5, 6, 7, 8], "test": [9, 11], "rigid_transform": False},
#     2: {"train": [1, 5, 6, 7, 8], "test": [9, 11], "rigid_transform": True},
# }
# self.protocol = protocols[protocol]

import sys

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.append('/home/datta/lab/HPE3D/src')


def plot_3d(pose):

    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    # ax._axis3don = False
    print(pose)
    x = pose[:, 0]
    # y = -1*pose[:, 2]
    # z = -1*pose[:, 1]

    y = pose[:, 2]
    z = pose[:, 1]

    ax.scatter(x, y, z)
    skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
    for link in skeleton:
        ax.plot(x[([link[0], link[1]])],
                y[([link[0], link[1]])], z[([link[0], link[1]])])

    labels = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
              'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')

    for i, j, k, l in zip(x, y, z, labels):
        ax.text(i, j, k, s=f'{l} | {int(i)} | {int(j)} | {int(k)}', size=7, zorder=1, color='k')


    xx, yy = np.meshgrid([0,-700], [-5480, -5080])
    zz = np.ones((len(xx), len(yy))) * min(z)*1.01  # padding
    ax.plot_surface(xx, yy, zz, cmap='gray',
                    linewidth=0, alpha=0.2)

    ax.axis = 'off'
    plt.savefig('/home/datta/lab/HPE3D/src/results/h363d.png', format='png', dpi=1000)
    plt.show()

    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)



def plot_2d(pose, image=None):

    fig = plt.figure(1)
    ax = fig.gca()
    if image:
        image = Image.open(image)
        ax.imshow(image)
    
    ax.set_xticks([])
    ax.set_yticks([])

    # plt.savefig('/home/datta/lab/HPE3D/src/results/h36image.png', format='png', dpi=1000)
    # plt.show()
    # return
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100]

    x = pose[:, 0]
    # y = -1*pose[:, 1]
    y = pose[:, 1]

    if image:
        # To plot 2d and image together
        # print(pose[i,:], end = "\t")
        pose[:, :] = pose[:,:] - pose[0] 
        pose[:,:] /= np.max(pose[:,:])
        pose[:,:] *= 256/2  
        # pose[:,0] += 131  
        # pose[:,1] += 117  
        
            # print(pose[i,:])
        x = pose[:, 0]
        y = pose[:, 1]


    ax.scatter(x, y)
    skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
    for link in skeleton:
        ax.plot(x[([link[0], link[1]])],
                y[([link[0], link[1]])])

    labels = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
              'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')

    for i, j, l in zip(x, y, labels):
        ax.text(i+2, j+2, s=l, size=8, color='k')

    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    ax.set_xticks([])
    ax.set_yticks([])

    ax.axis = 'off'
    # plt.title('2D Pose')

    plt.savefig('/home/datta/lab/HPE3D/src/results/2dpose.png', format='png', dpi=1000)
    plt.show()
    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)


if __name__ == "__main__":

    image_path = '/home/datta/lab/HPE_datasets/h36m/'
    save_path = '/home/datta/lab/HPE3D/src/results'

    h5name = '/home/datta/lab/HPE3D/src/data/debug_h36m17_911.h5'

    f = h5py.File(h5name, 'r')
    # print(list(f.keys()))

    i = 1

    pose3 = f['pose3d'][i]
    pose2 = f['pose2d'][i]
    subject_ = f['subject'][i]
    action_ = f['action'][i]
    subaction_ = f['subaction'][i]
    camera_ = f['camera'][i]
    idx = f['idx'][i]
    f.close()

    dirname = 's_%02d_act_%02d_subact_%02d_ca_%02d' % (
        subject_, action_, subaction_, camera_)
    image = image_path+dirname+"/"+dirname+"_"+("%06d" % (idx))+".jpg"
    # print(pose2.shape)
    # print(pose3.shape)
    # plot_2d(pose2)
    # plot_2d(pose2, image)
    # plot_h36(pose3)
    import viz
    viz.plot_2d(pose2)
    viz.plot_3d(pose3)
    import gc
    gc.collect()
