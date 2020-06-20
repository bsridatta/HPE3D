'''
Code adapted from dataset provided by https://github.com/juyongchang/PoseLifter
'''

import numpy as np
import os
import scipy.io as sio
import h5py

img_path = '/home/datta/lab/HPE_datasets/h36m/'
save_path = '/home/datta/lab/HPE3D/src/data/'
annot_name = 'matlab_meta_new.mat'

subject_list = [9, 11]
# subject_list = [1, 5, 6, 7, 8]
subj_str = "".join(str(x) for x in subject_list)
h5name = 'h36m17_' + subj_str + '.h5'

inds = range(17)
action_list = np.arange(2, 17)
subaction_list = np.arange(1, 3)
camera_list = np.arange(1, 5)

# to get smaller subset of the data for dev
debug = True

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

f = h5py.File(save_path+h5name, 'w')
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

print(f"Saved to {save_path+h5name}")