import os
import sys

import h5py
import numpy as np
import scipy.io as sio

# from src.data_preparation.camera_parameters import get_camera_data
###############################################################
# Set Paths
save_path = f'{os.getenv("HOME")}/lab/HPE3D/src/data/'
img_path = f'{os.getenv("HOME")}/lab/HPE_datasets/h36m_poselifter/'
annot_name = 'matlab_meta_new.mat'

# Set Annotations to retrieve
subject_list = [1]
skip_frame = 64

# subject_list = [1, 5, 6, 7, 8]
# skip_frame = 5

subj_str = "".join(str(x) for x in subject_list)
h5_save_name = 'h36m17_sh_' + subj_str
inds = range(17)
action_list = np.arange(2, 3)  # 17
subaction_list = np.arange(1, 3)
camera_list = np.arange(1, 5)

################################
#             SH               #

camera_ids = {1: "54138969",
              2: "55011271",
              3: "58860488",
              4: "60457274"
              }

sh_path = f'{os.getenv("HOME")}/lab/HPE_datasets/h36m_h5s/'

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



################################

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
posesh = []
pose2d = []
pose3d = []
pose3d_global = []
bbox = []
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
                annot_file = img_path + dir_name + '/' + annot_name
                try:
                    data = sio.loadmat(annot_file)
                    # print("done ", dir_name)
                except:
                    print('pass %s' % dir_name)
                    continue

                ########### SH ############
                sh_file_name = f'S{subject_}/StackedHourglass/{action_names[action_]}_0{subaction_}.{camera_ids[camera_]}'
                sh_annot_file = sh_path + sh_file_name + '.h5'
                try:
                    sh_data = h5py.File(sh_annot_file, 'r')
                except:
                    print('pass %s' % sh_file_name)
                    continue

                posesh_ = np.asarray(sh_data['poses'])[:, swap_inds] # reorder joints
                posesh_ = np.insert(posesh_, 0, np.zeros((1,2)), 1) # add root
                posesh_ = np.transpose(posesh_, (1,2,0))
                #########################

                pose2d_ = np.transpose(data['pose2d'], (2, 1, 0))
                pose3d_ = np.transpose(data['pose3d'], (2, 1, 0))
                pose3d_global_ = np.transpose(data['pose3d_global'], (2, 1, 0))
                bbox_ = data['bbox']

                num_images = pose2d_.shape[2]
                for i in range(num_images):
                    if i % skip_frame != 0:
                        continue
                    idx.append(i+1)
                    
                    ##### SH #####
                    print(inds, i, posesh_.shape, len(posesh), pose2d_.shape, len(pose2d))

                    posesh.append(posesh_[inds, :, i])
                    
                    pose2d.append(pose2d_[inds, :, i])
                    pose3d.append(pose3d_[inds, :, i])
                    pose3d_global.append(pose3d_global_[inds, :, i])
                    bbox.append(bbox_[i].astype(int))
                    subject.append(subject_)
                    action.append(action_)
                    subaction.append(subaction_)
                    camera.append(camera_)
                    num_samples += 1

                    #############################################################
                    # Comment or uncomment these debug conditions to change the debug dataset distribution

                    if debug:
                        break

                print(subject_, action_, pose2d_.shape, posesh_.shape)

                if debug:
                    break
            if debug:
                break
        # if debug:
        #     break
    # if debug:
    #     break

print(f'number of samples = %d' % num_samples)

exit("not saving")

f = h5py.File(save_path+h5_save_name+'.h5', 'w')
f['idx'] = idx
f['posesh'] = posesh
f['pose2d'] = pose2d
f['pose3d'] = pose3d
f['pose3d_global'] = pose3d_global
f['bbox'] = bbox
f['cam_f'] = cam_f
f['cam_c'] = cam_c
f['cam_R'] = cam_R
f['cam_T'] = cam_T
f['cam_p'] = cam_p
f['cam_k'] = cam_k
f['subject'] = subject
f['action'] = action
f['subaction'] = subaction
f['camera'] = camera
f.close()

print(f"Saved to {save_path+h5_save_name+'.h5'}")
