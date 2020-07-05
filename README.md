# ROS toolkit for deep feature extraction

This repo contains the following ROS packages:
- feature_extraction: real-time extraction of image features (keypoints and their descriptors and scores, and per-image global descriptors)
- image_feature_msgs: definition of feature messages

# Setup

### System requirement

- Ubuntu 18.04 + ROS Melodic (recommended version)
- Python 3.6 or higher
- TensorFlow 1.12 or higher (`pip3 install tensorflow`)
- (optinoal) OpenVINO 2019 R3 or higher ([download](https://software.intel.com/en-us/openvino-toolkit/choose-download))
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

2. Download cv_bridge and configure it for Python3 (required by feature_extraction for using cv_bridge in Python3)
```
git clone -b melodic https://github.com/ros-perception/vision_opencv.git
cd ..
# change the path in the following command according to your Python version
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```

3. Build
```
. /opt/ros/melodic/setup.bash
catkin build
```

4. Donwload one of the saved [HF-Net](https://github.com/ethz-asl/hfnet) models from [here](https://github.com/cedrusx/open_deep_features/releases/tag/model_release_1), and unzip it.

# Run

### Feature extraction

Start the feature extraction node, which will subscribe to one or more image topic(s) and publish the extracted image features on corresponding topic(s) with `/features` suffix.
```
. YOUR_PATH_TO_CATKIN_WS/devel/setup.bash
```

With OpenVINO model:
```
. /opt/intel/openvino/bin/setupvars.sh
rosrun feature_extraction feature_extraction_node.py _net:=hfnet_vino _model_path:=YOUR_PATH_TO/models/hfnet_vino_480x640
```

With TensorFlow model:
```
rosrun feature_extraction feature_extraction_node.py _net:=hfnet_tf _model_path:=YOUR_PATH_TO/models/hfnet_tf
```

Additional params and their default values:
```
_keypoint_number:=500 \
_gui=True \
```

