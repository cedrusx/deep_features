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
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test feature extraction and visualize keypoints')
    parser.add_argument('--net', type=str, default='hfnet_vino',
        help='Network model: hfnet_vino (default), hfnet_tf.')
    parser.add_argument('--pause', action='store_true',
        help='Wait for keyboard input after showing each image.')
    parser.add_argument('--no-gui', action='store_true',
        help='Do not visualize keypoints.')
    args,filenames = parser.parse_known_args()

    net_name = args.net
    try:
        cmd = 'from %s import FeatureNet, default_config' % (net_name)
        exec(cmd, globals())
    except ImportError as err:
        exit('Unknown net %s: %s' % (net_name, str(err)))
    config = default_config
    #config['keypoint_threshold'] = 0
    net = FeatureNet(config)
    frame_count = 0
    total_time = 0.
    # do an extra infer first because it may take much more time and mislead the performance analysis
    if len(filenames) > 0:
        net.infer(cv2.imread(filenames[0]))
    for f in filenames:
        image = cv2.imread(f)
        image = cv2.resize(image, (640, 480))
        start_time = time.time()
        features = net.infer(image)
        end_time = time.time()
        frame_time = (end_time - start_time) * 1000
        total_time += frame_time
        frame_count += 1
        num_keypoints = features['keypoints'].shape[0]
        print(f + ': ' + str(image.shape) +
                ', %d keypoints, %.2f ms (average %.2f ms)' % (num_keypoints, frame_time, total_time / frame_count))
        if args.no_gui: continue
        draw_keypoints(image, features['keypoints'], features['scores'])
        if args.pause:
            title = f + ' (' + net_name + ', ' + str(num_keypoints) + ' keypoints)'
            cv2.imshow(title, image)
            cv2.waitKey()
        else:
            title = net_name
            cv2.imshow(title, image)
            cv2.waitKey(1)

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