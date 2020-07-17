def plot_mayavi(pose):
    """plot single pose as save as wavefront

    Args:
        pose (numpy array): 17x3
    """
    black = (0, 0, 0)
    white = (1, 1, 1)

    mlab.figure(size=(1024, 768),
                bgcolor=(1, 1, 1), fgcolor=(1 ,1, 1))

    # joints and connections
    skeleton = ((0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12), (12, 13),
                (8, 14), (14, 15), (15, 16), (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6))

    # pose data specific
    x = pose[:, 0]
    y = -1*pose[:, 2]
    z = -1*pose[:, 1]

    pts = mlab.points3d(x, y, z, scale_factor=30, color=(1, 0, 0))
    pts.mlab_source.dataset.lines = np.array(skeleton)
    tube = mlab.pipeline.tube(pts, tube_radius=15)
    # tube.filter.radius_factor = 1.
    tube = mlab.pipeline.stripper(tube)
    mlab.pipeline.surface(tube, color=(1, 0.0, 0))

    # Fancy stuff
    # mlab.pipeline.surface(lines, colormap='Accent', line_width=1, opacity=.4)

    # And choose a nice view
    # mlab.view(330.6, 106, 6000.5, [0, 0, .05])
    # mlab.roll(0)

    mlab.savefig("pose.obj")
    mlab.show()


if __name__ == "__main__":
    
    import subprocess
    import sys
 
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mayavi"])
    
    import numpy as np
    from mayavi import mlab

    pose = [[0.0000,    0.0000,    0.0000],
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

    pose = np.asarray(pose)

    plot_mayavi(pose)
