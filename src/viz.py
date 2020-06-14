import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ACTION_NAMES = ["Directions", "Discussion", "Eating", "Greeting", "Phoning", "Photo", "Posing", "Purchases",
                "Sitting", "SittingDown", "Smoking", "Waiting", "WalkDog", "Walking", "WalkTogether"]


def plot_diff(ax, pose, target, error):
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

    # fig = plt.figure(1)
    # ax = fig.gca(projection='3d')
    ax._axis3don = False

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

    for i in range(0, rows*cols):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        ax = plot_diff(ax, poses[i].numpy(),
                       targets[i].numpy(), errors[i].item())

    plt.show()


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
