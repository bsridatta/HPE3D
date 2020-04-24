import numpy as np
from mayavi import mlab

def may():
    black = (0,0,0)
    white = (1,1,1)
    mlab.figure(bgcolor=white)
    mlab.plot3d([0, 1000], [0, 0], [0, 0], color=black, tube_radius=15.)


    # Finally, display the set of lines
    # mlab.pipeline.surface(lines, colormap='Accent', line_width=1, opacity=.4)

    # And choose a nice view
    mlab.view(33.6, 106, 5.5, [0, 0, .05])
    mlab.roll(125)
    mlab.show()

def tri():
    import trimesh
    mesh = trimesh.load('snapshot.obj')
    scene_split = trimesh.scene.split_scene(mesh)
    print(scene_split)

if __name__=="__main__":
    tri()
