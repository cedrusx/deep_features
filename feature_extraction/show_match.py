#!/usr/bin/env python3
"""
A non-ROS script to visualize extracted keypoints and their matches of given image pairs
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
    args,filenames = parser.parse_known_args()

    net_name = args.net
    gui = True
    if net_name == 'hfnet_vino':
        from hfnet_vino import FeatureNet, default_config
    elif net_name == 'hfnet_tf':
        from hfnet_tf import FeatureNet, default_config
    else:
        exit('Unknown net %s' % net_name)
    config = default_config
    #config['keypoint_threshold'] = 0.001
    net = FeatureNet(config)
    file_features = {}
    for f in filenames:
        image = cv2.imread(f)
        #image = cv2.resize(image, (640, 480))
        #cv2.imshow(f, image)
        start_time = time.time()
        features = net.infer(image)
        end_time = time.time()
        num_keypoints = features['keypoints'].shape[0]
        print(f + ': ' + str(image.shape) +
                ', %d keypoints, %.2f ms' % (num_keypoints, (end_time - start_time) * 1000))
        file_features[f] = features
        file_features[f]['image'] = image
        if gui:
            draw_keypoints(image, features['keypoints'], features['scores'])
            title = f + ' (' + net_name + ', ' + str(num_keypoints) + ' keypoints)'
            cv2.imshow(title, image)
            cv2.waitKey()

    f1 = filenames[0]
    for f2 in filenames[1:]:
        distance = np.linalg.norm(file_features[f1]['global_descriptor'] \
                                - file_features[f2]['global_descriptor'])
        des1 = list(file_features[f1]['local_descriptors'])
        des2 = list(file_features[f2]['local_descriptors'])
        des1 = np.squeeze(file_features[f1]['local_descriptors'])
        des2 = np.squeeze(file_features[f2]['local_descriptors'])
        kp1 = [cv2.KeyPoint(int(p[0]), int(p[1]), _size=2) for p in file_features[f1]['keypoints']]
        kp2 = [cv2.KeyPoint(int(p[0]), int(p[1]), _size=2) for p in file_features[f2]['keypoints']]
        img1 = file_features[f1]['image']
        img2 = file_features[f2]['image']

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        #matches = sorted(matches, key = lambda x:x.distance)
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
        title = os.path.splitext(os.path.basename(f1))[0] + '-' + \
                os.path.splitext(os.path.basename(f2))[0] + '-' + str(distance)
        cv2.imshow(title, match_img)
        cv2.imwrite(title + '.jpg', match_img)
        cv2.waitKey()

def draw_keypoints(image, keypoints, scores):
    upper_score = 0.5
    lower_score = 0.1
    scale = 1 / (upper_score - lower_score)
    for p,s in zip(keypoints, scores):
        s = min(max(s - lower_score, 0) * scale, 1)
        color = (255 * (1 - s), 255 * (1 - s), 255) # BGR
        cv2.circle(image, tuple(p), 3, color, 2)


if __name__ == "__main__":
    main()