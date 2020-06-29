'''
Code adapted from dataset provided by https://github.com/juyongchang/PoseLifter

-- Use this file to extract annotations as h5 files, view samples of the data without additional preprocessing
(only includes processing done by Pose lifter)

-- Use to create mean and variance of the 2D/3D pose per joint or across the whole dataset
'''

import os

import h5py
import numpy as np
import scipy.io as sio

###############################################################
# Set Paths
img_path = f'{os.getenv("HOME")}/lab/HPE_datasets/h36m/'
save_path = f'{os.getenv("HOME")}/lab/HPE3D/src/data/'
annot_name = 'matlab_meta_new.mat'

# Set Annotations to retrieve
subject_list = [9, 11]
# subject_list = [1, 5, 6, 7, 8]
subj_str = "".join(str(x) for x in subject_list)
h5name = 'h36m17_' + subj_str
inds = range(17)
action_list = np.arange(2, 17)
subaction_list = np.arange(1, 3)
camera_list = np.arange(1, 5)

# Get smaller subset of the data for fast dev?
debug = True
# Get Mean and Std of the data alone?
mean_std = False

#################################################################

if debug:
    h5name = "debug_" + h5name

if not os.path.exists(save_path):
    os.mkdir(save_path)

idx = []
pose2d = []
pose3d = []
pose3d_global = []
bbox = []
cam_f = []
cam_c = []
cam_R = []
cam_T = []
subject = []
action = []
subaction = []
camera = []
num_samples = 0

for subject_ in subject_list:
    for action_ in action_list:
        for subaction_ in subaction_list:
            for camera_ in camera_list:
                dir_name = 's_%02d_act_%02d_subact_%02d_ca_%02d' % (
                    subject_, action_, subaction_, camera_)
                print(dir_name)
                annot_file = img_path + dir_name + '/' + annot_name
                try:
                    data = sio.loadmat(annot_file)
                except:
                    print('pass %s' % dir_name)
                    continue
                pose2d_ = np.transpose(data['pose2d'], (2, 1, 0))
                pose3d_ = np.transpose(data['pose3d'], (2, 1, 0))
                pose3d_global_ = np.transpose(data['pose3d_global'], (2, 1, 0))
                bbox_ = data['bbox']
                cam_f_ = data['f']
                cam_c_ = data['c']
                cam_R_ = data['R']
                cam_T_ = data['T']
                num_images = pose2d_.shape[2]
                for i in range(num_images):
                    if i % 5 != 0:
                        continue
                    idx.append(i+1)
                    pose2d.append(pose2d_[inds, :, i])
                    pose3d.append(pose3d_[inds, :, i])
                    pose3d_global.append(pose3d_global_[inds, :, i])
                    bbox.append(bbox_[i].astype(int))
                    cam_f.append(cam_f_[i])
                    cam_c.append(cam_c_[i])
                    cam_R.append(cam_R_[i])
                    cam_T.append(cam_T_[i])
                    subject.append(subject_)
                    action.append(action_)
                    subaction.append(subaction_)
                    camera.append(camera_)
                    num_samples += 1

                    print(subject_, action_, pose3d_.shape)

                    #############################################################
                    # Comment or uncomment these debug conditions to change the debug dataset distribution

                    if debug:
                        break
                if debug:
                    break
            if debug:
                break
        # if debug:
        #     break
    # if debug:
    #     break

print(f'{dir_name}  number of samples = %d' % num_samples)

if mean_std:
    import json

    norm_stats = {}
    pose2d = np.asarray(pose2d)
    norm_stats['mean2d'] = str(np.mean(pose2d, axis=(0,1)))
    norm_stats['mean3d'] = str(np.mean(pose3d, axis=(0,1)))
    norm_stats['std2d'] = str(np.std(pose2d, axis=(0,1)))
    norm_stats['std3d'] = str(np.std(pose3d, axis=(0,1)))

    norm_stats['max2d'] = str(np.max(pose2d, axis=(0,1)))
    norm_stats['max3d'] = str(np.max(pose3d, axis=(0,1)))

    norm_stats['min2d'] = str(np.min(pose2d, axis=(0,1)))
    norm_stats['min3d'] = str(np.min(pose3d, axis=(0,1)))
    
    with open(save_path+h5name+"_mean_std.json", 'w') as f:
        json.dump(norm_stats, f)
    # Using only h36m17_15678_mean/std//2d/3d.csv 

    exit()

f = h5py.File(save_path+h5name+'.h5', 'w')
f['idx'] = idx
f['pose2d'] = pose2d
f['pose3d'] = pose3d
f['pose3d_global'] = pose3d_global
f['bbox'] = bbox
f['cam_f'] = cam_f
f['cam_c'] = cam_c
f['cam_R'] = cam_R
f['cam_T'] = cam_T
f['subject'] = subject
f['action'] = action
f['subaction'] = subaction
f['camera'] = camera
f.close()

print(f"Saved to {save_path+h5name+'.h5'}")
