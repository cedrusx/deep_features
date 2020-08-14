# ROS toolkit for deep feature extraction

This repo contains the following ROS packages:
- feature_extraction: real-time extraction of image features (keypoints and their descriptors and scores, and per-image global descriptors)
- image_feature_msgs: definition of feature messages

There are also non-ROS scripts for feature extraction, keypoint visualization and keypoint matching, which do not require ROS installation nor the building procedures in below:
- feature_extraction/show_keypoints.py: show extracted keypoints from given image files
- feature_extraction/show_match.py: show feature matching results from give image pairs

# Setup

### System requirement

- Ubuntu + ROS
- Python 3.6 or higher
- TensorFlow 1.12 or higher (`pip3 install tensorflow`)
- OpenVINO 2020 R1 or higher ([download](https://software.intel.com/en-us/openvino-toolkit/choose-download))
- OpenCV for Python3 (`pip3 install opencv-python`; not needed if OpenVINO is installed and activated)
- numpy (`pip3 install numpy`)
- No GPU requirement

### Download and build

0. Preliminary
```
sudo apt install python3-dev python-catkin-tools python3-catkin-pkg-modules python3-rospkg-modules python3-empy python3-yaml
```

1. Set up catkin workspace and download this repo
```
mkdir src && cd src
git clone https://github.com/cedrusx/deep_features_ros.git
```

2. To use `cv_bridge` in Python3 with ROS Melodic or older versions, you need to compile it locally. NOT needed if you are using ROS Noetic or newer version.
```
git clone -b melodic https://github.com/ros-perception/vision_opencv.git
```

3. Build
```
# go to your catkin workspace (the parent folder of src)
cd ..
# configure to build for Python3 - Please change the path in the following command according to your Python version
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
# set the parent catkin workspace
. /opt/ros/melodic/setup.bash
# build with catkin from the python-catkin-tools package
catkin build
```

5. Donwload one of the saved [HF-Net](https://github.com/ethz-asl/hfnet) models from the [Releases](https://github.com/cedrusx/deep_features/releases), and unzip it.

# Run

### Feature extraction

Start the feature extraction node, which will subscribe to one or more image topic(s) and publish the extracted image features on corresponding topic(s) with `/features` suffix.

With OpenVINO model:
```
. /opt/intel/openvino/bin/setupvars.sh
. YOUR_PATH_TO_CATKIN_WS/devel/setup.bash
rosrun feature_extraction feature_extraction_node.py _topics:=/YOUR_CAMERA_TOPIC _net:=hfnet_vino _model_path:=YOUR_PATH_TO_MODEL_FOLDER
```

With TensorFlow model:
```
. YOUR_PATH_TO_CATKIN_WS/devel/setup.bash
rosrun feature_extraction feature_extraction_node.py _topics:=/YOUR_CAMERA_TOPIC _net:=hfnet_tf _model_path:=YOUR_PATH_TO_MODEL_FOLDER
```

The `topics` param can take one or more topic names (separated by a comma), e.g. `/usb_cam/image_raw`, or `/left_cam/image_raw,/right_cam/image_raw`.

Additional params and their default values (more model-specific params are defined in the `default_config` dict in the source code):
```
_keypoint_number:=500 \
_keypoint_threshold:=0.002 \
_gui:=True \
_log_interval:=3.0 \
```

