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

def get_samples(self, idx=1):
    image_path = f'{os.getenv("HOME")}/lab/HPE_datasets/h36m/'
    h5name = f'{os.getenv("HOME")}/lab/HPE3D/src/data/h36m17_911.h5'
    f = h5py.File(h5name, 'r')

    sample = {}
    for x in f.keys():
        sample[x] = f[x][idx]

    sample["pose3d_noise"] = sample["pose3d"].copy()
    sample["pose3d_noise"][0, 2] = 1  # noise

    dirname = 's_%02d_act_%02d_subact_%02d_ca_%02d' % (
        sample.subject, sample.action, sample.subaction, sample.camera)
    image = image_path+dirname+"/"+dirname+"_"+("%06d" % (sample.idx))+".jpg"
