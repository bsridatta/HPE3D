import io
import math
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

ACTION_NAMES = ["Directions", "Discussion", "Eating", "Greeting", "Phoning", "Photo", "Posing", "Purchases",
                "Sitting", "SittingDown", "Smoking", "Waiting", "WalkDog", "Walking", "WalkTogether"]

SKELETON_COLORS = ['b', 'b', 'b', 'b', 'orange', 'orange', 'orange',
                   'b', 'b', 'b', 'b', 'b', 'b', 'orange', 'orange', 'orange', 'orange']


def plot_diff(ax, pose, target, error, i, image=True):
    """plot the prediction and ground with error

    Arguments:
        ax {axes} -- empty matplotlib axes
        pose {tensor/array} -- single prediction
        target {tensor/array} -- single ground truth
        error {float} -- average pjpe of the prediction

    Returns:
        ax -- ax with plot
    """
    # info to connect joints
    skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
    labels = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
              'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')

    fig = plt.figure(1)
    # ax = fig.gca(projection='3d')
    # ax._axis3don = False
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    print("POSE")
    x = pose[:, 0]
    y = -1*pose[:, 2]
    z = -1*pose[:, 1]

    x1 = target[:, 0]
    y1 = -1*target[:, 2]
    z1 = -1*target[:, 1]

    ax.scatter(x, y, z, alpha=0.6, s=2)
    ax.scatter(x1, y1, z1, c='grey', s=1, alpha=0.7)

    verts = []

    for link in skeleton:
        ax.plot(x[([link[0], link[1]])],
                y[([link[0], link[1]])], z[([link[0], link[1]])],
                c='b', alpha=0.6, lw=3)

        ax.plot(x1[([link[0], link[1]])],
                y1[([link[0], link[1]])], z1[([link[0], link[1]])],
                c='grey', alpha=0.6, lw=1)

        area = [(x[link[0]], y[link[0]], z[link[0]]),
                (x[link[1]], y[link[1]], z[link[1]]),
                (x1[link[1]], y1[link[1]], z1[link[1]]),
                (x1[link[0]], y1[link[0]], z1[link[0]])
                ]

        verts.append(area)

    ax.add_collection3d(Poly3DCollection(verts, facecolors=[
                        'r', 'r'], alpha=0.2, zorder='max'))

    # Show coordinate values
    # for i, j, k, l in zip(x, y, z, labels):
    #     ax.text(i, j, k, s=l, size=8, zorder=1, color='k')

    # Plot the surface.
    xx, yy = np.meshgrid([-500, 500], [-500, 500])
    zz = np.ones((len(xx), len(yy))) * min(min(z), min(z1))*1.01  # padding
    ax.plot_surface(xx, yy, zz, cmap='gray',
                    linewidth=0, alpha=0.2)

    ax.text(200, 200, zz[0][0], s=str(round(error, 2)) +
            " mm", size=8, zorder=1, color='k')

    ax2 = fig.add_subplot(1, 2, 2)
    root = f"{os.getenv('HOME')}/lab/HPE_datasets/h36m"
    path = "s_11_act_02_subact_01_ca_02"
    image = mpimg.imread(f'{root}/{path}/{path}_'+"%06d" % (5*i+1)+".jpg")
    ax2.imshow(image)

    fig.savefig(f"{os.getenv('HOME')}/lab/HPE3D/src/results/{i}.png")
    fig.clf()

    return ax

    # animate rotation of pose
    # plt.show()
    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(.001)


def plot_diffs(poses, targets, errors, grid=5):
    """Calls plot_diff and plots the diff between all poses and targets

    Arguments:
        poses {list} -- list of prediction
        targets {list} -- list of ground truths
        errors {list} -- list of avg pjpe for each prediction

    Keyword Arguments:
        grid {int} -- number of plots to show (default: {5})
    """
    fig = plt.figure(figsize=(15., 12.))
    plt.rcParams['savefig.dpi'] = 300

    rows = cols = grid  # math.ceil(math.sqrt(len(poses)))
    from tqdm import tqdm

    # for i in range(0, rows*cols):
    for i in tqdm(range(0, poses.shape[0], 10)):
        # ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        print(i)
        ax = 0
        plot_diff(ax, poses[i].numpy(),
                  targets[i].numpy(), errors[i].item(), i)

    # plt.show()


def plot_umap(zs, metrics):
    """plot UMAP

    Arguments:
        zs {list} -- list of latent embeddings of 2D poses (/images)
        metrics {list} -- metrics to color the embeddings say, actions
    """
    import umap
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("[INFO]: UMAP reducing ", [*zs.shape])
    reducer = umap.UMAP(n_neighbors=3,
                        min_dist=0.1,
                        metric='cosine')
    embedding = reducer.fit_transform(zs)
    print('[INFO]: Embedding shape ', embedding.shape)

    sns.scatterplot(x=embedding[:, 0],
                    y=embedding[:, 1],
                    hue=[ACTION_NAMES[int(x-2)] for x in metrics.tolist()],
                    palette="Set2",
                    alpha=0.6)

    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
    plt.axis('tight')
    # plt.gca().set_aspect( 'datalim') #'equal',
    plt.title(f'UMAP projection of Z  {[*zs.shape]}', fontsize=15)
    plt.show()


def plot_3d(pose, return_image=True, axis=None):
    """plot the prediction and ground with error

    Arguments:
        pose {array} -- single numpy array if joints 16 - for root addition
                        else tensor also works as
    """
    # info to connect joints
    skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
    labels = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
              'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')

    if axis is None:
        fig = plt.figure(1)
        ax = fig.gca(projection='3d')
        # ax._axis3don = False
    else:
        ax = axis

    if pose.shape[0] == 16:
        pose = np.concatenate((np.zeros((1, 3)), pose), axis=0)

    x = pose[:, 0]
    y = -1*pose[:, 2]
    z = -1*pose[:, 1]

    ax.scatter(x, y, z, alpha=0.6, s=0.1)

    for link, color in zip(skeleton, SKELETON_COLORS):
        ax.plot(x[([link[0], link[1]])],
                y[([link[0], link[1]])],
                z[([link[0], link[1]])],
                c=color, alpha=0.6, lw=3)

    # From https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-tohttps://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array(
        [x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -
                                1:2:2][0].flatten() + 0.5*(x.max()+x.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -
                                1:2:2][1].flatten() + 0.5*(y.max()+y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2, -1:2:2, -
                                1:2:2][2].flatten() + 0.5*(z.max()+z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

    for i, j, k, l in zip(x, y, z, labels):
        ax.text(i, j, k, s=f'{l}', size=7, zorder=1, color='k')
        #  | {int(i)} | {int(j)} | {int(k)}

    # plt.tight_layout()
    # ax.set_aspect('equal')
    # ax.tick_params(which='minor',top=False, bottom=False, left=False, right=False,
    #             labelleft=False, labelbottom=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if axis:
        return ax

    plt.show()

    if return_image:
        """for latent viz"""
        DPI = fig.get_dpi()
        fig.set_size_inches(305.0/float(DPI), 305.0/float(DPI))
        fig.savefig(f"{os.getenv('HOME')}/lab/HPE3D/src/results/x.png")
        fig.clf()
        image = Image.open(f"{os.getenv('HOME')}/lab/HPE3D/src/results/x.png")
        image = image.convert('RGB')
        # image = np.asarray(image)
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # to RGB or image = image[...,0:3]
        image = transforms.ToTensor()(image).unsqueeze_(0)

        return image

    """
    # Show coordinate values
    for i, j, k, l in zip(x, y, z, labels):
        ax.text(i, j, k, s=l, size=8, zorder=1, color='k')

    # Plot the surface.
    xx, yy = np.meshgrid([-max(x), max(x)], [-max(y), max(y)])
    zz = np.ones((len(xx), len(yy))) * min(z)*1.01  # padding
    ax.plot_surface(xx, yy, zz, cmap='gray',
                    linewidth=0, alpha=0.2)

    # trick to make aspect ratio equal
    ### Make axes limits
    xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
    XYZlim = [min(xyzlim[0]), max(xyzlim[1])]
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)
    try:
        ax.set_aspect('equal')
    except NotImplementedError:
        pass
    
    
    # animate rotation of pose
    plt.show()
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.001)
    """

def plot_2d(pose, image=False, axis=None):
    """plot the prediction and ground with error

    Arguments:
        pose {array} -- single numpy array if joints 16 - for root addition
                        else tensor also works as

    """
    # info to connect joints
    skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
    labels = ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck',
              'Nose', 'Head', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist')

    if axis == None:
        fig = plt.figure(1)
        ax = fig.gca()
    else:
        ax = axis

    if pose.shape[0] == 16:
        pose = np.concatenate((np.zeros((1, 2)), pose), axis=0)

    x = pose[:, 0]
    y = -1*pose[:, 1]

    ax.scatter(x, y, alpha=0.6, s=2)

    for link, color in zip(skeleton, SKELETON_COLORS):
        ax.plot(x[([link[0], link[1]])],
                y[([link[0], link[1]])],
                c=color, alpha=0.6, lw=3)

    for i, j, l in zip(x, y, labels):
        ax.text(i, j, s=l, size=8, zorder=1, color='k')

    ax.set_aspect('equal')

    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    if axis:
        return ax

    plt.show()


def decode_embedding(config, model):
    decoder = model[1]
    decoder.eval()
    with torch.no_grad():
        samples = torch.randn(10, 30).to(config.device)
        samples = decoder(samples)
        if '3D' in decoder.__class__.__name__:
            samples = samples.reshape([-1, 16, 3])
        elif 'RGB' in decoder.__class__.__name__:
            samples = samples.reshape([-1, 256, 256])
        # TODO save as images to tensorboard


def plot_mayavi(pose, pose2):

    import numpy as np
    from mayavi import mlab

    black = (0, 0, 0)
    white = (1, 1, 1)
    # mlab.figure(bgcolor=black)
    mlab.figure(size=(1024, 768),
                bgcolor=(1, 1, 1), fgcolor=(0.5, 0.5, 0.5))

    skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))

    x = pose[:, 0]
    y = -1*pose[:, 2]
    z = -1*pose[:, 1]

    pts = mlab.points3d(x, y, z, scale_factor=30, color=(1, 0, 0))
    pts.mlab_source.dataset.lines = np.array(skeleton)
    tube = mlab.pipeline.tube(pts, tube_radius=15)
    # tube.filter.radius_factor = 1.
    tube = mlab.pipeline.stripper(tube)
    mlab.pipeline.surface(tube, color=(1, 0.0, 0))

    pose = pose2
    x = pose[:, 0]
    y = -1*pose[:, 2]
    z = -1*pose[:, 1]

    pts = mlab.points3d(x, y, z, scale_factor=30, color=(1, 1, 1))
    pts.mlab_source.dataset.lines = np.array(skeleton)
    tube = mlab.pipeline.tube(pts, tube_radius=15)
    # tube.filter.radius_factor = 1.
    tube = mlab.pipeline.stripper(tube)
    mlab.pipeline.surface(tube, color=(1, 1, 1))

    # mlab.pipeline.surface(lines, colormap='Accent', line_width=1, opacity=.4)

    # And choose a nice view
    # mlab.view(330.6, 106, 5.5, [0, 0, .05])
    # mlab.roll(125)
    mlab.savefig("stick_2.obj")
    mlab.show()


def plot_pose(pose2d=None, pose3d=None, image=None):
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
        plot_2d(pose2d, axis=ax)

    if pose3d is not None:
        ax = fig.add_subplot(100+col*10+i, projection='3d')
        i += 1
        plot_3d(pose3d, axis=ax)

    plt.show()


if __name__ == "__main__":

    pose3d = np.asarray([[0.0000,    0.0000,    0.0000],
                         [122.7085,  -17.2441,   42.9420],
                         [126.0797,  444.2065,  119.1129],
                         [155.8211,  903.9439,  107.8988],
                         [-122.7145,   17.2554,  -42.7849],
                         [-138.7586,  479.9395,   19.2924],
                         [-106.0115,  940.2942,    5.0193],
                         [12.2478, -243.5484,  -50.2997],
                         [22.3039, -479.3382, -106.1938],
                         [11.8855, -534.7589,  -60.3629],
                         [33.6124, -643.4368, -119.8231],
                         [-127.0257, -429.1896, -193.8714],
                         [-384.9372, -379.3297, -305.7618],
                         [-627.3461, -393.2285, -330.2295],
                         [188.2248, -445.5474,  -52.9106],
                         [445.9429, -379.3877,   54.3570],
                         [641.9531, -382.0340,  210.9446]]
                        )
    pose3d_error = np.asarray([[0.0000,    0.0000,    1.0000],
                               [122.7085,  -17.2441,   42.9420],
                               [126.0797,  444.2065,  119.1129],
                               [158.8211,  904.9439,  108.8988],
                               [-122.7145,   17.2554,  -42.7849],
                               [-138.7586,  479.9395,   19.2924],
                               [-106.0115,  940.2942,    5.0193],
                               [12.2478, -243.5484,  -50.2997],
                               [22.3039, -479.3382, -106.1938],
                               [11.8855, -534.7589,  -60.3629],
                               [33.6124, -643.4368, -119.8231],
                               [-127.0257, -429.1896, -193.8714],
                               [-384.9372, -379.3297, -305.7618],
                               [-627.3461, -393.2285, -330.2295],
                               [188.2248, -445.5474,  -52.9106],
                               [445.9429, -379.3877,   54.3570],
                               [641.9531, -382.0340,  210.9446]]
                              )
    # plot_mayavi(np.asarray(pose), np.asarray(pose2))
    plot_3d(np.asarray(pose3d))
    # plot_pose(pose3d=pose3d)
