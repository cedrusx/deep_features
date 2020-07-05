#!/usr/bin/env python3
"""
A non-ROS script to visualize extracted keypoints of given images
"""

import os
import cv2
import numpy as np
import time
import threading
import sys

def main():
    net_name = 'hfnet_vino'
    gui = True
    if net_name == 'hfnet_vino':
        from hfnet_vino import FeatureNet, default_config
    elif net_name == 'hfnet_tf':
        from hfnet_tf import FeatureNet, default_config
    else:
        exit('Unknown net %s' % net_name)
    config = default_config
    #config['keypoint_threshold'] = 0
    net = FeatureNet(config)
    filenames = sys.argv[1:]
    for f in filenames:
        image = cv2.imread(f)
        image = cv2.resize(image, (640, 480))
        start_time = time.time()
        features = net.infer(image)
        end_time = time.time()
        num_keypoints = features['keypoints'].shape[0]
        print(f + ': ' + str(image.shape) +
                ', %d keypoints, %.2f ms' % (num_keypoints, (end_time - start_time) * 1000))
        if gui:
            draw_keypoints(image, features['keypoints'], features['scores'])
            title = f + ' (' + net_name + ', ' + str(num_keypoints) + ' keypoints)'
            cv2.imshow(title, image)
            cv2.waitKey()

def draw_keypoints(image, keypoints, scores):
    upper_score = 0.2   # keypoints with this score or higher will have a red circle
    lower_score = 0.002 # keypoints with this score or lower will have a white circle
    scale = 1 / (upper_score - lower_score)
    for p,s in zip(keypoints, scores):
        s = min(max(s - lower_score, 0) * scale, 1)
        color = (255 * (1 - s), 255 * (1 - s), 255) # BGR
        cv2.circle(image, tuple(p), 3, color, 1)


if __name__ == "__main__":
    main()