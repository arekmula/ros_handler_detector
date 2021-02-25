# ros_handler_detector

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

## Run with

- Setup path to your model directory and label map:
```
rosparam set model_dir "path/to/model"
rosparam set label_map_path "path/to/labelmap"
```
- Setup RGB image (640x480) topic:
```
rosparam set rgb_image_topic "image/topic"
```

- Run with
```
rosrun handler_detector handler_detector.py
```