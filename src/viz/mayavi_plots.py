from mayavi import mlab
import numpy as np

def plot_3D_models(poses, mode="show"):
    """plot single pose as save as wavefront

    Args:
        pose (numpy array): 16/17 joints
    """
    black = (0, 0, 0)
    white = (1, 1, 1)

    mlab.figure(size=(1024, 768),
                bgcolor=(1, 1, 1), fgcolor=(1, 1, 1))

    # joints and connections
    skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))
    colors = [(1, 0, 0), (1, 1, 1)]

    for pose, color in zip(poses, colors[0:len(poses)]):
        if pose.shape[0] == 16:
            pose = np.concatenate((np.zeros((1, 3)), pose), axis=0)

        # pose data specific
        x = pose[:, 0]
        y = -1*pose[:, 2]
        z = -1*pose[:, 1]

        pts = mlab.points3d(x, y, z, scale_factor=30, color=(1, 0, 0))
        pts.mlab_source.dataset.lines = np.array(skeleton)
        tube = mlab.pipeline.tube(pts, tube_radius=15)
        # tube.filter.radius_factor = 1.
        tube = mlab.pipeline.stripper(tube)
        mlab.pipeline.surface(tube, color=color)

        # Fancy stuff
        # mlab.pipeline.surface(lines, colormap='Accent', line_width=1, opacity=.4)

    # And choose a nice view
    # mlab.view(330.6, 106, 6000.5, [0, 0, .05])
    # mlab.roll(0)
    
    if mode == 'save':
        mlab.savefig("/lhome/sbudara/lab/HPE3D/src/results/pose.obj")
        mlab.close()
        return 
    
    # else:
    #    mlab.show()