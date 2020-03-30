# Human Pose Estimation

1. [3D HPE](#3D-HPE)
2. [Datasets](#Datasets)
3. [More Stuff](#More-Papers)  

3D HPE
======

#### Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image   
[[Paper](https://arxiv.org/pdf/1907.11346v2.pdf)]
[[Code-Pose](https://github.com/mks0601/3DMPPE_POSENET_RELEASE)] 
[[Code-Root](https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE)] 
[**ICCV 2019**]

<details>
<summary>
absolute pose, multi person, pose depth, cascade
</summary>  
  
> * A general framework with 3 networks. 1). Human detection 2). RootNet - Human root localization in global 3D world. 3). PoseNet - 3D single-person pose w.r.t Root. where, Root is a fixed ref. point of human body say, pelvis.  
> * The RootNet learns 2D co-ordinates of the root and depth separately. Depth is estimated using metric euivlent to pinhole camera model by assuming a constant area for a human in real world and area in image world, i.e bounding box after converting to 1:1 aspect ratio. To handle this constant area assumption (i.e if the human is child or in sitting postion etc), the RootNet from image features, learns a parameter to rectify the area in image world instead to nullify the error in area in real world. Thus predicting a good depth estimate of the human root
> * Using the localized root from RootNet and 3D pose relative to root from PoseNet, all the detected 3D poses are shifted to respective positions in the global 3D world. As it is modular one can replace the models with SOTA to improve performance. 

</details>

#### Unsupervised 3D Pose Estimation with Geometric Self-Supervision 
[[Paper](https://arxiv.org/pdf/1904.04812.pdf)]
[**CVPR 2019**]

<details>
<summary>
self supervised, discriminator, domain adaptation, temporal consistency    
</summary>
  
> * Unsupervised learning to lift 2D joints to 3D skeletons 
> * Lifter network outputs 3D pose which is then rotated in random angles and is projected to 2D in a different POV. A discriminator is used to evaluate if this new 2D pose is in the possible pose distribution which is learnt from 2D pose datasets.
> * Geometric Self Consistency  
    1. Since rotations should not change a 3D pose, this new 2D projection when lifted again should give a 3D skeleton when rotated back to the original POV gives back the original 3D pose.   
    2. And the re-projection of this new 3D skeleton that is rotated to original POV should give a 2D joint identical to the initial 2D joints. These geometric consistencies can be used to generate large data in a self-supervised manner. 
> * Since training unsupervisedly need more data, a 2D adapter network is trained to convert 2D joints from source domain to a target domain
> * For sequential 2D pose from video, temporal discriminator is used to evaluate if 2D pose is real or fake based on the previous 2D pose during run time. This improves performance in inference time even when input is not sequential.   

</details>


#### AbsPoseLifter: Absolute 3D Human Pose Lifting Network from a Single Noisy 2D Human Pose
[[Paper](https://arxiv.org/pdf/1910.12029.pdf)]
[2019]
</br>
<details>
<summary>
absolute pose, noise, errors, cascade    
</summary>

> * method for estimating the root coordinates and root-relative 3D pose simultaneously 
> * Ambiguities also applies to other papers
    1. size ambiguity - the size of the human subject is learned implicitly from datasets
    2. focal length ambiguity - outputs the canonical root depth normalized by the focal length instead of the real depth. If additional focal length information is available, we can obtain the root’s real depth from the canonical depth 
> * 2D pose estimation errors exhibit a similar distribution regardless of the type of 2D pose estimator
> * In order to train the lifting network such a distribution of errors are imparted to the 2D GT to synthasize realistic 2D pose that a 2D pose estimator would provide
> * novel normalization layer normalizes the input 2D pose and adds the target subject’s 2D location and scale information as intermediate features.

</details>

#### RepNet: Weakly Supervised Training of an Adversarial Reprojection Network for 3D Human Pose Estimation
[[Paper](https://arxiv.org/pdf/1902.09868.pdf)]
[[Code](https://github.com/bastianwandt/RepNet)]
[**CVPR 2019**]
<details>
<summary>
weakly supervised, adversial training, GAN    
</summary>

> * An adversarial training method for a 3D human pose
estimation neural network (RepNet) based on a 2D reprojection
> * Weakly supervised training without 2D-3D correspondences and unknown cameras.
> * Simultaneous 3D skeletal keypoints and camera pose
estimation
> * A layer encoding a kinematic chain representation that
includes bone lengths and joint angle informations
> * A pose regression network that generalizes well to unknown human poses and cameras
</details>

#### Distill Knowledge from NRSfM for Weakly Supervised 3D Pose Learning
[[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Distill_Knowledge_From_NRSfM_for_Weakly_Supervised_3D_Pose_Learning_ICCV_2019_paper.pdf)]
[**ICCV 2019**]
<details>
<summary>
teacher student, weakly supervised, NRSfM    
</summary>
  
> * Weakly supervised pose estimation method using solely 2D landmark annotations  
> * Strong NRSfM baseline modified from Deep-NRSfM, which outperforms current published state-of-the-art NRSfM methods on H3.6M dataset  
> * New knowledge distilling algorithm applicable to NRSfM methods based on dictionary learning. Demonstrates that our learned network gets significantly lower error on the training set compared to its NRSfM teacher

</details>


#### Unsupervised Adversarial Learning of 3D Human Pose from 2D Joint Locations
[[Paper](https://arxiv.org/pdf/1803.08244.pdf)]
[[Code](https://github.com/DwangoMediaVillage/3dpose_gan)]
[2018]
<details>
<summary>
GAN, unsupervised    
</summary>

> * An unsupervised method that learns a 3D human pose from 2D joint locations in a single image without any 3D datasets
> * One of the first unsupervised paper?
</details>


#### C3DPO: Canonical 3D Pose Networks for Non-Rigid Structure From Motion 
[[Paper](https://arxiv.org/pdf/1909.02533.pdf)]
[[Code](https://github.com/facebookresearch/c3dpo_nrsfm)]
[**ICCV 2019**]
</br>
<details>
<summary>
canonicalization network, NRSFM, self-supervised
</summary>

> * recovers both 3D canonical shape and viewpoint using only 2D keypoints in a single image at test time
> * uses a novel self-supervised constraint "canoniacalization network" to correctly factorize 3D shape and viewpoint
> * can handle occlusions and missing values in the observations
> * works effectively across multiple object categories
> * both the factorization network and canonicalization network share the same core architecture
> * losses - 1) reprojection loss 2) canonicalization network loss 3) rotation invariance loss 
</details>



<!-- Template for a paper
#### Title 
[[Paper](https://arxiv.org/pdf/)]
[**Venue**]
</br>
<details>
<summary>
Keyword1, keyword2, keyword3    
</summary>
<*Remove this*/br>  
> * Keypoint 1 
> * Keypoint 2
</details>
End of Template -->  

Concepts
========
* Reprojection Error Optimization - 3D human body
model is deformed such that it satisfies a reprojection error  
* Direct NN inference -  estimate 3D poses directly from images or detected
keypoints
* Kinematic Chain Space (KCS) - Projecting 3D human pose into KCS, a contraint is derived that is based on the assumption that the bone lengths are constant. This can be benificial as giving an additional feature matrix to the network, it doesnt have to learn joint length computation and angular constraints on its own. [Ref Paper](https://arxiv.org/pdf/1702.00186.pdf)
* Non rigid structure from motion (NRSfM) - aims to obtain the varying 3D structure and camera motion from uncalibrated 2D point tracks [NR-SFM video explanation](https://www.youtube.com/watch?v=zBalNj2F8Ik)


Datasets
========
* [Human3.6](http://vision.imar.ro/human3.6m/description.php)[2014] - 2D and 3D relative joint pose, multi view
* [MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)[2017] - 2D and 3D relative joint pose, mutli view
* [JTA Joint Track Auto](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=25)[2018] - 2D, 3D, occulusion annotation, synthetic

More Stuff
===========
* [LEarning to Reconstruct people](https://sric.me/Learning-to-Reconstruct-People/)
* [Gyeongsik Moon - Seoul National University](https://scholar.google.com.hk/citations?user=2f2D258AAAAJ&hl=zh-CN)
* [awesome-human-pose-estimation](https://github.com/wangzheallen/awesome-human-pose-estimation)
* [3d-human-pose-estimation](https://github.com/trumDog/3d-human-pose-estimation)
