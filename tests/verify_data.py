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

import os
import h5py
from src.viz.mpl_plots import plot_data, plot_errors, plot_3d, plot_2d, plot_superimposition
from src.viz.mayavi_plots import plot_3D_models
from src.dataset import H36M
from src.processing import preprocess


def get_raw_sample(i):
    image_path = f'{os.getenv("HOME")}/lab/HPE_datasets/h36m_poselifter/'
    h5name = f'{os.getenv("HOME")}/lab/HPE3D/src/data/h36m17_911.h5'
    # h5name = f'{os.getenv("HOME")}/lab/HPE3D/src/data/h36m17_5frame_911.h5'

    f = h5py.File(h5name, 'r')

    sample = {}
    for x in f.keys():
        sample[x] = f[x][i]

    dirname = 's_%02d_act_%02d_subact_%02d_ca_%02d' % (
        sample['subject'], sample['action'], sample['subaction'], sample['camera'])
    image = image_path+dirname+"/"+dirname + \
        "_"+("%06d" % (sample['idx']))+".jpg"
    sample['image'] = image

    f.close()

    return sample


def get_processed_sample(i):

    annotation_file = f'h36m17_5frame'
    image_path = f"{os.getenv('HOME')}/lab/HPE_datasets/h36m_poselifter/"

    dataset = H36M([9, 11],
                   annotation_file, image_path, train=True, no_images=False, projection=True)

    print("[INFO]: Length of the dataset: ", len(dataset))
    print("[INFO]: One sample -")

    sample = dataset.__getitem__(i)

    for key in sample.keys():
        sample[key] = sample[key].numpy()

    del dataset
    return sample
        
def main(idx, plot):
    plot = plot
    processed = False
    # idx = 33987
    if processed:
        sample = get_processed_sample(idx)
    else:
        sample = get_raw_sample(idx)

    image = sample['image']
    pose2d = sample['pose2d']
    pose3d = sample['pose3d']

    sample["pose3d_noise"] = sample["pose3d"].copy()
    sample["pose3d_noise"][0, 0] = 1  # noise

    # 2D, 3D, Image
    if plot == 1:
        # plot_data( pose3d=pose3d)
        plot_data(image=image, pose2d=pose2d, pose3d=pose3d)
    # 3D Model
    elif plot == 2:
        print(pose3d)
        plot_3D_models([pose3d])
    # 3D Model diff
    elif plot == 3:
        plot_3D_models([pose3d, sample["pose3d_noise"]])
    elif plot == 4:
        img = plot_3d(pose3d, color='blue', mode="show", show_ticks=True, labels=False, 
        title=str(min(pose3d[:,2]))+" "+str(max(pose3d[:,2]))) 
    # MPL Grid diff
    elif plot == 5:
        plot_errors(poses=[pose3d]*20, targets=[sample['pose3d_global']]*20, errors= [2]*20)
        # plot_errors(poses=[pose3d, pose3d], targets=[pose3d_noise, pose3d_noise])
    # Zeroed 3D Model
    elif plot == 6:
        plot_3D_models([pose3d-pose3d[0]])
    # MPL projection
    elif plot == 7:
        from src.viz.mpl_plots import plot_projection_raw
        plot_projection_raw(sample)
    # plot rotation
    elif plot == 8:
        from src.processing import random_rotate_and_project_3d_to_2d
        import torch
        import math
        pose3d = torch.tensor(pose3d)
        # pose3d = torch.index_select(pose3d, -1, torch.tensor([1,0,2]))
        pose3d = torch.stack((pose3d, pose3d), axis=0)
        while 1:
            rot = random_rotate_and_project_3d_to_2d(
                pose3d,
                roll_range=(-math.pi / 6.0,
                            math.pi / 6.0),
                azimuth_range=(0, 0),
                elevation_range=(-math.pi, math.pi),
                default_camera=True,
                default_camera_z=10.0,
                random_rotate=True)
            # UNCOMMENT PROJ FOR THIS TO WORK, ONLY ROTATE
            plot_3d(pose3d[0].numpy(), color='gray', mode="axis",
                    show_ticks=True, labels=False, mean_root=True)
            plot_3d(rot[0].numpy(), color='blue', mode="show",
                    show_ticks=True, labels=False, mean_root=True)

            # Azimuth roll (roll book wrt normal)
            # Roll elevation (rotate wrt horizontal line)
            # Elevation Azimuth (rotate wrt vertical line)
    elif plot==9:
        from src.processing import random_rotate, procrustes
        import torch
        import math

        pose3d = torch.tensor(pose3d)

        # pose3d = torch.index_select(pose3d, -1, torch.tensor([1,0,2]))
        # pose3d = torch.stack((pose3d, pose3d), axis=0)
        # gt = pose3d
        # # rotate and scale and translate
        # inp = random_rotate(gt)
        # inp = torch.tensor((0,0,1000)) + inp
        # # pose3d = torch.tensor((0,0,-1000)) + pose3d

        # out = procrustes(gt, inp, allow_scaling=False, allow_reflection=True)

        # plot_3d(gt[0].numpy(), color='gray', mode="axis",
        #         show_ticks=True, labels=False, mean_root=True)
        # plot_3d(inp[0].numpy(), color='green', mode="axis",
        #         show_ticks=True, labels=False, mean_root=True)
        # plot_3d(out[0].numpy(), color='orange', mode="show",
        #                 show_ticks=True, labels=False, mean_root=True)        
        
        out= pose3d 
        plot_3d(out.numpy(), color=None, mode="show",
                show_ticks=False, labels=False, mean_root=False, axis3don=False)
    
    if plot == 10:
        plot_superimposition(pose2d, image, sample['bbox'], filename=f"{os.getenv('HOME')}/lab/HPE3D/src/results/{idx}_inp.png")

    if plot == 11:
        plot_2d(pose2d, mode='image', filename=f"{os.getenv('HOME')}/lab/HPE3D/src/results/{idx}_2d.png", save=True)

if __name__ == "__main__":
    # means
    ids = [3493,6422,14027,74176,21621,26692,30089,33987,39416,94650,96348,54005,55495,106863,109473]
    # fails
    ids = [2878,6130,73047,18120,24801,26722,31669,34930,90950,43450,98756,101351,105002,106047,61221]
    # neighbours
    ids = [7817, 7742,7123,6102,7752,5484,3418,2850,6441,3279,7194,7126,7126,7264,7254,7386,7237,7108,7043,3079,2895,3125,2888,2909,791,5617,6755,826,629,8435,5850,8051,7614,7924,516,8070]    
    # ids = [3418, 2848, 6441, 3279, 2185, 817, ]
    # ids = [7817]
    # for i in ids:
    #     print(i)
    # main(i, plot=11)
    main(0, plot=11)