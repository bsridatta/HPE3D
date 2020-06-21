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

sys.path.append('/home/datta/lab/HPE3D/src')


def plot_h36(pose):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure(1)
    ax = fig.gca(projection='3d')

    n = 100

    # create a 21 x 21 vertex mesh
    xx, yy = np.meshgrid(np.linspace(0, 1, 21), np.linspace(0, 1, 21))

    # create vertices for a rotated mesh (3D rotation matrix)
    X = xx
    Y = yy
    Z = 10*np.ones(X.shape)

    # create some dummy data (20 x 20) for the image
    data = np.cos(xx) * np.cos(xx) + np.sin(yy) * np.sin(yy)
    ax.contourf(X, Y, data, 0, zdir='z', offset=0.5, cmap=cm.BrBG)

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    x = pose[:, 0]
    y = -1*pose[:, 2]
    z = -1*pose[:, 1]

    ax.scatter(x, y, z)
    skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
    for link in skeleton:
        ax.plot(x[([link[0], link[1]])],
                y[([link[0], link[1]])], z[([link[0], link[1]])])

    labels = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
              'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')

    for i, j, k, l in zip(x, y, z, labels):
        ax.text(i, j, k, s=l, size=8, zorder=1, color='k')

    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    ax.axis = 'off'

    plt.show()

    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)


def plot_3D(pose):
    x = pose[:, 0]
    y = -1*pose[:, 2]
    z = -1*pose[:, 1]
    x = x/x.max()
    y = y/y.min()
    z = z/z.min()
    skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))

    fig = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=4,
            color=z,
            colorscale='Viridis'
        ),
        showlegend=False
    ))

    for link in skeleton:
        fig.add_trace(go.Scatter3d(
            x=x[([link[0], link[1]])],
            y=y[([link[0], link[1]])],
            z=z[([link[0], link[1]])],
            mode='lines',
            showlegend=False
        ))
    fig.update_layout(scene=dict(
        xaxis_title='X AXIS TITLE',
        yaxis_title='Y AXIS TITLE',
        zaxis_title='Z AXIS TITLE'),
        width=700,
        margin=dict(r=20, b=10, l=10, t=10))

    fig.show()


def may(pose):
    import numpy as np
    from mayavi import mlab
    black = (0, 0, 0)
    white = (1, 1, 1)
    mlab.figure(bgcolor=white)

    x = pose[:, 0]
    y = -1*pose[:, 2]
    z = -1*pose[:, 1]
    skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
    i = 0
    for link in skeleton:
        x1 = [x[link[0]], x[link[1]]]
        y1 = [y[link[0]], y[link[1]]]
        z1 = [z[link[0]], z[link[1]]]
        mlab.plot3d(x1, y1, z1, color=black, tube_radius=10.)
        if i == 1:
            break
        i += 1

    # Finally, display the set of lines
    # mlab.pipeline.surface(lines, colormap='Accent', line_width=1, opacity=.4)

    # And choose a nice view
    mlab.view(33.6, 106, 5.5, [0, 0, .05])
    mlab.roll(125)
    mlab.savefig("stick_2.obj")
    mlab.show()


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
    img_file = image_path+dirname+"/"+dirname+"_"+("%06d" % (idx))+".jpg"
    print(pose2.shape)
    print(pose3.shape)
    import viz
    viz.plot_2d(pose2)
    
    # plot_h36(pose3)
    import gc
    gc.collect()
