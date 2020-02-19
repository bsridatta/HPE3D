# Human Pose Estimation

1. [3D HPE](#3D-HPE)
2. [Datasets](#Datasets)
3. [More Papers](#More-Papers)  

3D HPE
======

#### Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image   
[[Paper](https://arxiv.org/pdf/1907.11346v2.pdf)]
[[Code-Pose](https://github.com/mks0601/3DMPPE_POSENET_RELEASE)] 
[[Code-Root](https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE)] 
[**ICCV 2019**]

<details>
<summary>
absolute pose, multi person, pose depth   
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


#### Unsupervised Adversarial Learning of 3D Human Pose from 2D Joint Locations
[[Paper](https://arxiv.org/pdf/1803.08244.pdf)]
[[Code](https://github.com/DwangoMediaVillage/3dpose_gan)]
[**CVPR 2018**]
</br>
<details>
<summary>
GAN, unsupervised    
</summary>
</br>  
> * An unsupervised method that learns a 3D human pose from 2D joint locations in a single image without any 3D datasets
> * Keypoint 2
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
</br>  
> * Keypoint 1 
> * Keypoint 2
</details>
End of Template -->  


Datasets
========

More Papers
===========

* [Gyeongsik Moon - Seoul National University](https://scholar.google.com.hk/citations?user=2f2D258AAAAJ&hl=zh-CN)
* [awesome-human-pose-estimation](https://github.com/wangzheallen/awesome-human-pose-estimation)
* [3d-human-pose-estimation](https://github.com/trumDog/3d-human-pose-estimation)
