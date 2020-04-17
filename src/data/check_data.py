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


import numpy as np
import h5py
import pandas as pd
import plotly.graph_objects as go
import skimage.io as sio
from PIL import Image

def plot_h36(pose):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure(1)
    ax = fig.gca(projection='3d')

    n = 100


    # create a 21 x 21 vertex mesh
    xx, yy = np.meshgrid(np.linspace(0,1,21), np.linspace(0,1,21))

    # create vertices for a rotated mesh (3D rotation matrix)
    X =  xx 
    Y =  yy
    Z =  10*np.ones(X.shape)

    # create some dummy data (20 x 20) for the image
    data = np.cos(xx) * np.cos(xx) + np.sin(yy) * np.sin(yy)
    ax.contourf(X, Y, data, 0, zdir='z', offset=0.5, cmap=cm.BrBG)

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    x =  pose[:,0]
    y =  -1*pose[:,2]
    z =  -1*pose[:,1]
    
    ax.scatter(x, y, z)
    skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
    for link in skeleton:
        ax.plot(x[([link[0], link[1]])], y[([link[0], link[1]])], z[([link[0], link[1]])])

    labels = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')

    for i,j,k, l in zip(x,y,z,labels):
        ax.text(i, j, k , s = l, size=8, zorder=1, color='k')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)
    
if __name__ == "__main__":

    image_path = '../../../HPE_datasets/h36m/'
    save_path = './'

    h5name = save_path + 'debug_h36m17.h5'

    f = h5py.File(h5name, 'r')
    train_f = {}
    train_subjects = [1,5,6,7,8]
    train_indices = []
    for i, subj in enumerate(f['subject']):
        if subj in train_subjects: 
            train_indices.append(i)
    print(f['subject'])
    for k in f.keys():
        f[k] = f[k][train_indices]
    print(f['subject'])
    
    exit()
    i = 10

    pose = f['pose3d'][i]
    subject_ = f['subject'][i]
    action_ = f['action'][i]
    subaction_ = f['subaction'][i]
    camera_ = f['camera'][i]
    idx = f['idx'][i]
    f.close()
    dirname = 's_%02d_act_%02d_subact_%02d_ca_%02d' % (subject_, action_, subaction_, camera_)
    img_file = image_path+dirname+"/"+dirname+"_"+("%06d"%(idx))+".jpg"
    image = sio.imread(img_file)
    plot_h36(pose)
    import gc
    gc.collect()
