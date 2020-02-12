# awesome-HPE

### 3D HPE

Camera Distance-aware Top-down Approach for 3D Multi-person Pose Estimation from a Single RGB Image   
[[Paper](https://arxiv.org/pdf/1907.11346v2.pdf)]
[[Code-Pose](https://github.com/mks0601/3DMPPE_POSENET_RELEASE)] 
[[Code-Root](https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE)] 
[**ICCV 2019**]
  

> * A general framework with 3 networks. 1). Human detection 2). RootNet - Human root localization in global 3D world. 3). PoseNet - 3D single-person pose w.r.t Root. where, Root is a fixed ref. point of human body say, pelvis.  
> * The RootNet learns 2D co-ordinates of the root and depth separately. Depth is estimated using metric euivlent to pinhole camera model by assuming a constant area for a human in real world and area in image world, i.e bounding box after converting to 1:1 aspect ratio. To handle this constant area assumption (i.e if the human is child or in sitting postion etc), the RootNet from image features, learns a parameter to rectify the area in image world instead to nullify the error in area in real world. Thus predicting a good depth estimate of the human root
> * Using the localized root from RootNet and 3D pose relative to root from PoseNet, all the detected 3D poses are shifted to respective positions in the global 3D world. As it is modular one can replace the models with SOTA to improve performance. 

Abosulte Pose * Multi Person * Multi Dataset * Depth
##


[arxiv]: https://img.shields.io/badge/arXiv-lightgrey 
[github]: https://img.shields.io/badge/GitHub-code-lightgrey
