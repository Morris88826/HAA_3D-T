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

Recently added:

+baseball_pitch

+basketball_dribble

+basketball_layup

+bowing_waist

+brushing_floor

+play_leaf-flute

+play_lute

+play_maracas

+play_melodic

+play_noseflute

+play_ocarina

---

+CPR

+basketball_shoot

+beer_pong_throw

+bench_dip

+bending_back

+burpee

+play_timpani

+play_triangle

+play_trombone

+play_trumpet

+play_ukelele

+play_voila

+play_suona

---

+play_sanxian

+play_saw

+play_saxophone

+play_serpent

+play_sheng

+play_sitar

+play_tambourine

+play_thereminvox


## Annotation Tool

### Preparation
1. Download the RGB images of HAA500 and put it under the './dataset/raw' folder. 

2. Create the environment and install packages from requirements.txt. Conda version: 4.8.3
```
conda create --name haa python=3.7
conda activate haa
pip install numpy
pip install -r requirements.txt
```

### Get Started

Run the ui.py file. 

<img width="1155" alt="Screen Shot 2021-06-23 at 12 37 05 AM" src="https://user-images.githubusercontent.com/32810188/122966526-053b0b00-d3bc-11eb-8bf5-edb34406f22e.png">
1. Double click the video in the list to enter the annotating page. * means that the annotation is finished. (number of files in joints2d == number of RGB file)
<img width="1280" alt="Screen Shot 2021-07-05 at 5 00 58 PM" src="https://user-images.githubusercontent.com/32810188/124446033-a198e080-ddb2-11eb-9359-b03fe32f8dbe.png">


2. In the annotating page, users can use mouse scroll and drag to move the frame into the wanted position. Click [Reset Zoom] button on the right to go back to the default.
3. To change the frame, use the bar below, or press (z) or (x) on your keyboard to go to the prev/next frame. Scroll the bar quickly for a video-like preview.
4. Users can press (Load from Alphapose) or (Load from joints2d) to load joints. For Alphapose, it will process to Evoskeleton style. If the alphapose information doesn't exist, it will run the AlphaPose network to generate the info file (make sure that your device has cuda). 
5. To label. You can press (l) on your keyboard to start labeling 17 joints. You can also use your keyboard to label a single joint. keys are (4),(5),(q-u), (a-j),(v). You can use the image on the right for the guide. Note that pink is the left arm. Click on the top of the image for labeling. If you want to delete a joint, press (2). If want to delete all joints in a frame, press (0).
6. One can label a joint to be occluded/non-visible or non-occluded/visible by toggling (1). In label-everything mode (l), pressing (1) will change the joint to be non-visible, while pressing (3) will mark it visible. If one press (`), it will toggle the visibility of a joint throughout all frames. Users can also press (Mark all unoccluded/occluded) to change all joints' visibility.
7. Press (o) to let all the joints in the current frame as user-selected (will be used for auto interpolation).
8. The tool supports the linear interpolation method by click (Auto interpolate). If you do not like the interpolation for some frames, you can just label correctly on those frames. For those joints that are interpolated, it is marked as visible.
9. We also provide a temporal network for the tool to estimate the position of the non-visible joints. Users can label only parts of the human body and press (temporal prediction) to select the frames they want to estimate. It not only provides the 2D position it also predicts the skeleton's position in 3D. A more detailed explanation of the temporal network can be found in https://github.com/Morris88826/HAA_temporalNet
10. Users can change the size of the dot and width of the bones in the UI for better visualization.
11. After users are done annotating all the frames, use motion smoothing to perform gaussian smoothing with sigma 1.
12. The saved skeletal data will be stored in './dataset/joints2d' and './dataset/joints3d' respectively.


## References
[1] Fang HS, Xie S, Tai YW, Lu C. Rmpe: Regional multi-person pose estimation. In Proceedings of the IEEE International Conference on Computer Vision, 2017 (pp. 2334-2343).

[2] Shichao Li, Lei Ke, Kevin Pratama, Yu-Wing Tai, Chi-KeungTang, and Kwang-Ting Cheng. Cascaded deep monocular 3dhuman pose estimation with evolutionary training data.


