import os
import sys

import h5py
import numpy as np
# from src.data_preparation.camera_parameters import get_camera_data
###############################################################
# Set Paths
data_path = f'{os.getenv("HOME")}/lab/HPE_datasets/h36m_h5s/'
save_path = f'{os.getenv("HOME")}/lab/HPE3D/src/data/'

camera_ids = {1: "54138969",
              2: "55011271",
              3: "58860488",
              4: "60457274"
              }

swap_inds = np.array([2,1,0,3,4,5,6,7,8,9,13,14,15,12,11,10])
# a[:,swap_ind]

# 2-17
action_names = {2: "Directions",
                3: "Discussion",
                4: "Eating",
                5: "Greeting",
                6: "Phoning",
                7: "Photo",
                8: "Posing",
                9: "Purchases",
                10: "Sitting",
                11: "SittingDown",
                12: "Smoking",
                13: "Waiting",
                14: "WalkDog",
                15: "Walking",
                16: "WalkTogether"}


# Set Annotations to retrieve
subject_list = [9, 11]
skip_frame = 64

# subject_list = [1, 5, 6, 7, 8]
# skip_frame = 5

subj_str = "".join(str(x) for x in subject_list)
h5_save_name = 'h36m17_sh_' + subj_str
inds = range(16)
action_list = np.arange(2, 17)  # 17
subaction_list = np.arange(1, 4)
camera_list = np.arange(1, 5)

############ subaction are weird not _1,_2 but "",_1,_2
subact_suff = {1:"_1",
                2:"_2",
                3: ""}

# Get smaller subset of the data for fast dev?
debug = False
# Get Mean and Std of the data alone?
mean_std = False

#################################################################

if debug:
    h5_save_name = "debug_" + h5_save_name

if not os.path.exists(save_path):
    os.mkdir(save_path)

idx = []
pose2d = []
subject = []
action = []
subaction = []
camera = []
num_samples = 0

for subject_ in subject_list:
    for action_ in action_list:
        for subaction_ in subaction_list:
            for camera_ in camera_list:
                dir_name = f'S{subject_}/StackedHourglass/'
                file_name = f'S{subject_}/StackedHourglass/{action_names[action_]}{subact_suff[subaction_]}.{camera_ids[camera_]}'
                print(file_name)
                annot_file = data_path + file_name + '.h5'
                try:
                    data = h5py.File(annot_file, 'r')
                except:
                    print('pass %s' % file_name)
                    continue

                pose2d_ = np.asarray(data['poses'])[:, swap_inds] # reorder joints
                pose2d_ = np.transpose(pose2d_, (1,2,0))
                # pose3d_ = np.transpose(data['pose3d'], (2, 1, 0))
                # pose3d_global_ = np.transpose(data['pose3d_global'], (2, 1, 0))
                # bbox_ = data['bbox']
                # cam_f_ = data['f']
                # cam_c_ = data['c']
                # cam_R_ = data['R']
                # cam_T_ = data['T']
                # cam_p_ = get_camera_data(camera_, subject_, 'p')
                # cam_k_ = get_camera_data(camera_, subject_, 'k')
                num_images = pose2d_.shape[2]
                for i in range(num_images):
                    if i % skip_frame != 0:
                        continue
                    idx.append(i+1)
                    pose2d.append(pose2d_[inds, :, i])
                    # pose3d.append(pose3d_[inds, :, i])
                    # pose3d_global.append(pose3d_global_[inds, :, i])
                    # bbox.append(bbox_[i].astype(int))
                    # cam_f.append(cam_f_[i])
                    # cam_c.append(cam_c_[i])
                    # cam_R.append(cam_R_[i])
                    # cam_T.append(cam_T_[i])
                    # cam_p.append(cam_p_)
                    # cam_k.append(cam_k_)
                    subject.append(subject_)
                    action.append(action_)
                    subaction.append(subaction_)
                    camera.append(camera_)
                    num_samples += 1

                    #############################################################
                    # Comment or uncomment these debug conditions to change the debug dataset distribution

                    if debug:
                        break

                print(subject_, action_, pose2d_.shape)

                if debug:
                    break
            if debug:
                break
        # if debug:
        #     break
    # if debug:
    #     break

print(f'number of samples = %d' % num_samples)

f = h5py.File(save_path+h5_save_name+'.h5', 'w')
f['idx'] = idx
f['pose2d'] = pose2d
# f['pose3d'] = pose3d
# f['pose3d_global'] = pose3d_global
# f['bbox'] = bbox
# f['cam_f'] = cam_f
# f['cam_c'] = cam_c
# f['cam_R'] = cam_R
# f['cam_T'] = cam_T
# f['cam_p'] = cam_p
# f['cam_k'] = cam_k
f['subject'] = subject
f['action'] = action
f['subaction'] = subaction
f['camera'] = camera
f.close()

print(f"Saved to {save_path+h5_save_name+'.h5'}")
