'''
protocol #1 - using all 4 camera views in subjects S1, S5, S6 and S7 for training and the 
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

import numpy as np
import h5py
import pandas as pd
import plotly.graph_objects as go
import skimage.io as sio
from PIL import Image

def surface(image, d):

    image = np.flipud(image)
    d = -1 * np.ones((image.shape[0], image.shape[1]))* z.min()
    print(d.shape)
    img_as_8bit = lambda x: np.array(Image.fromarray(x).convert('P', palette='WEB', dither=None))
    dum_img = Image.fromarray(np.ones((3,3,3), dtype='uint8')).convert('P', palette='WEB')
    idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))
    colorscale=[[i/255.0, "rgb({}, {}, {})".format(*rgb)] for i, rgb in enumerate(idx_to_color)]
    trace=go.Surface(
            z=d,
            surfacecolor=img_as_8bit(image),
            cmin=0, 
            cmax=255,
            colorscale=colorscale
            )
    return trace

def plot_3D(pose, image):


    x =  pose[:,0]     
    y =  -1*pose[:,2]  
    z =  -1*pose[:,1]  
    x = x/x.max()
    y = y/y.min()
    z = z/z.min()
    skeleton = ( (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6) )
    
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

    fig.add_trace(go.Image(z = image))

    fig.update_layout(scene = dict(
                        xaxis_title='X AXIS TITLE',
                        yaxis_title='Y AXIS TITLE',
                        zaxis_title='Z AXIS TITLE'),
                        width=700,
                        margin=dict(r=20, b=10, l=10, t=10))

    fig.show()

def plot17j(pose):
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

    img_path = '/home/datta/lab/HPE_datasets/human36m/'
    save_path = './'

    h5name = save_path + 'h36m17.h5'

    f = h5py.File(h5name, 'r')

    i = 10

    pose = f['pose3d'][i]
    subject_ = f['subject'][i]
    action_ = f['action'][i]
    subaction_ = f['subaction'][i]
    camera_ = f['camera'][i]
    camera_ = f['camera'][i]
    idx = f['idx'][i]
    f.close()
    path = '../../../HPE_datasets/h36m/'
    dirname = 's_%02d_act_%02d_subact_%02d_ca_%02d' % (subject_, action_, subaction_, camera_)
    img_path = path+dirname+"/"+dirname+"_"+("%06d"%(idx))+".jpg"
    # imgdata = imread(img_path)
    image = sio.imread(img_path)
    plot_3D(pose, image)
    import gc
    gc.collect()
