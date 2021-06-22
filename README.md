# "HAA102" Action Recognition Dataset

## Overview
We create a new 3D+T Human Action dataset that is more challenging for predicting poses than the current Action Recognition datasets. 

Most current Human Action Recognition datasets are created in lab environments and are thus too easy for current state-of-the-art recognition networks. In contrast, our dataset consists of real-life human action videos with corresponding per-frame 3D joint locations of the person. 

The human action video dataset that we use to generate the 3D skeletal frames on and create our dataset is HAA500. HAA500 is a video dataset of atomic human action video clips that provides more diversified poses with variation and examples closer to real-life activities.

Significantly, our dataset consists of two challenging cases: Occluded joints (both self-occlusion and that by other objects) and out-of-bound joints (joints of the person that are outside the frame boundaries). 

## Structure of the dataset
HAA102 currently contains 102 action classes and 2039 video samples. For each sample, it contains:
* RGB videos
* 2D and 3D skeletal data

The RGB videos are the raw images from HAA500. 

2D skeletal data can be generated from AlphaPose[1] or by human labeling. For human labeling, we provided an annotation tool with interpolation techniques that can faster the annotating process. The shape of the 2D skeletal data is (num_joints, 3), with dimension one being the (x, y, confidence_score) of a joint. If a joint is present in the image or not hidden, the confidence score will be 1, else -1. A more detailed topology of the skeletal is shown in Figure 1. 

For 3D skeletal data, we use a 3D lifting tool to lift the 2D joints to 3D, which is implemented based on the open-source EvoSkeleton[2]. The shape of the 3D skeletal data is (num_joints, 4), with dimension one being the (x, y, z, confidence_score) of a joint.

<p align="center">
  <img width="400"  src="https://user-images.githubusercontent.com/32810188/122911842-4d413a00-d38a-11eb-8af6-b167504927a1.png" />
</p>
 <p align="center"> Figure 1. HAA 3D+T Skeleton Topology</p>

## Action Classes

Here we provide the action classes that are currently labeled. New classes will be added throughout time.
<p align="center">
  <img src="https://i.imgur.com/M8A8D7N.png" />
</p>

## Annotation Tool


## References
[1] Fang HS, Xie S, Tai YW, Lu C. Rmpe: Regional multi-person pose estimation. In Proceedings of the IEEE International Conference on Computer Vision, 2017 (pp. 2334-2343).

[2] Shichao Li, Lei Ke, Kevin Pratama, Yu-Wing Tai, Chi-KeungTang, and Kwang-Ting Cheng. Cascaded deep monocular 3dhuman pose estimation with evolutionary training data.


