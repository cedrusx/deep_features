# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

std_msgs/Header header          # must have the same stamp with corresponding image message
std_msgs/Bool sorted_by_score   # whether the keypoints are sorted in descending order of their scores
image_feature_msgs/KeyPoint[] keypoints
float32[] scores       # score of each keypoint, must be either in the same size with keypoints or empty
std_msgs/Float32MultiArray descriptors  # local descriptors of keypoints
float32[] global_descriptor    # global descriptor of the full image
