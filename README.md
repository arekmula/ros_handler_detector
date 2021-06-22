# ros_handler_detector
## About
![](imgs/detections1.jpg =160x120)
![](imgs/detections2.jpg =160x120)
![](imgs/detections3.jpg =160x120)

The goal of the project is to build a ROS node that would be responsible for detecting handlers of articulated objects such as cabinets, wardrobes, or lockers. The module uses a neural network to perform the task and utilizes CenterNet Resnet50 V1 architecture. The dataset used for training, evaluation, and testing is available [here](https://drive.google.com/file/d/11P2eSMlXDSz065TxQTR-hYyBDkFpOWnZ/view?usp=sharing)

This module is part of my master thesis "Point cloud-based model of the scene enhanced with information about articulated objects" and works best with the other three modules that can be found here:
- [Front detector](https://github.com/arekmula/ros_front_detection_segmentation)
- [Rotational joint detector](https://github.com/arekmula/ros_joint_segmentation)
- [Articulated objects scene builder](https://github.com/arekmula/articulated_objects_scene_builder)

## Results
- mAP@IoU=.50 -> **0.928**
- mAP@IoU=.75 -> **0.473**
- mAP@IoU=0.50:0.95 -> **0.503**

## Prerequisities
- Ubuntu 20.04
- ROS Noetic
- Tensorflow 2
- Python 3.8

## Installation
First ROS Noetic and tensorflow should be installed
Then:
```
mkdir -p caktin_ws/src
cd catkin_ws
catkin_make
cd src
git clone https://github.com/arekmula/ros_handler_detector
cd ros_handler_detector/src
protoc object_detection/protos/*.proto --python_out=.
cd ~/catkin_ws
rosdep install --from-path src/ -y -i
catkin_make
```

## Run 

- Setup path to your model directory and label map:
```
rosparam set model_dir "path/to/model"
rosparam set label_map_path "path/to/labelmap"
```
- Setup RGB image (640x480) topic:
```
rosparam set rgb_image_topic "image/topic"
```
- Determine if visualization image should be published
```
rosparam set visualize_handler_prediction True/False
```

- Run with
```
rosrun handler_detector handler_detector.py
```