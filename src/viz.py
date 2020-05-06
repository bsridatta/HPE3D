from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def plot_h36(pose):

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

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.axis = 'off'

    plt.show()
