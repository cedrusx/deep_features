#!/usr/bin/env python3

# Copyright (C) <2020-2021> Intel Corporation
# SPDX-License-Identifier: MIT

import os
import cv2
import numpy as np
import time
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from image_feature_msgs.msg import ImageFeatures, KeyPoint
from std_msgs.msg import MultiArrayDimension
import threading
import queue

def main():
    rospy.init_node('feature_extraction_node')
    #========================================= ROS params =========================================
    # Available nets: hfnet_vino, hfnet_tf
    net_name = rospy.get_param('~net', 'hfnet_tf')
    # User can set more than one input image topics, e.g. /cam1/image,/cam2/image
    topics = rospy.get_param('~topics', '/d400/color/image_raw')
    # Set gui:=True to pop up a window for each topic showing detected keypoints,
    # which will also be published to corresponding keypoints topics (e.g. /cam1/keypoints)
    gui = rospy.get_param('~gui', False)
    # For every log_interval seconds we will print performance stats for each topic
    log_interval = rospy.get_param('~log_interval', 3.0)
    #==============================================================================================

    if net_name == 'hfnet_vino':
        from hfnet_vino import FeatureNet, default_config
    elif net_name == 'hfnet_tf':
        from hfnet_tf import FeatureNet, default_config
    else:
        exit('Unknown net %s' % net_name)
    config = default_config
    #====================================== More ROS params =======================================
    # Model path and hyperparameters are provided by the specifed net by default, but can be changed
    # e.g., one can tell the desired (maximal) numbers of keypoints by setting keypoint_number,
    #       or filtering out low-quality keypoints by setting a high value of keypoint_threshold
    for item in config.keys():
        config[item] = rospy.get_param('~' + item, config[item])
    #==============================================================================================
    net = FeatureNet()
    node = Node(net, gui, log_interval)
    for topic in topics.split(','):
        node.subscribe(topic)
    rospy.spin()

class Node():
    def __init__(self, net, gui, log_interval):
        self.net = net
        self.gui = gui
        self.log_interval = log_interval
        self.cv_bridge = CvBridge()
        self.feature_publishers = {}
        self.keypoint_publishers = {}
        self.subscribers = {}
        self.latest_msgs = {}
        self.latest_msgs_lock = threading.Lock()
        self.stats = {}
        self.stats_lock = threading.Lock()
        self.result_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.worker)
        self.worker_thread.start()
        self.publisher_thread = threading.Thread(target=self.publisher)
        self.publisher_thread.start()

    def subscribe(self, topic):
        base_topic = '/'.join(topic.split('/')[:-1])
        self.feature_publishers[topic] = rospy.Publisher(base_topic + '/features', ImageFeatures, queue_size=1)
        self.keypoint_publishers[topic] = rospy.Publisher(base_topic + '/keypoints', Image, queue_size=1)
        self.stats[topic] = {'received': 0, 'processed': 0, 'last_time': None}
        with self.latest_msgs_lock:
            self.latest_msgs[topic] = None
        callback = lambda msg: self.callback(msg, topic)
        self.subscribers[topic] = rospy.Subscriber(topic, Image, callback, queue_size=1, buff_size=2**24)

    def callback(self, msg, topic):
        # keep only the lastest message
        with self.latest_msgs_lock:
            self.latest_msgs[topic] = msg
        with self.stats_lock:
            self.stats[topic]['received'] += 1

    def worker(self):
        while not rospy.is_shutdown():
            no_new_msg = True
            # take turn to process each topic
            for topic in self.latest_msgs.keys():
                with self.latest_msgs_lock:
                    msg = self.latest_msgs[topic]
                    self.latest_msgs[topic] = None
                if msg is not None:
                    self.process(msg, topic)
                    with self.stats_lock:
                        self.stats[topic]['processed'] += 1
                    no_new_msg = False
                self.print_stats(topic)
            if no_new_msg: time.sleep(0.01)

    def publisher(self):
        while not rospy.is_shutdown():
            if self.result_queue.qsize() > 5:
                rospy.logwarn_throttle(1, 'WOW! Inference is faster than publishing' +
                    ' (%d unpublished result in the queue)\n' % self.result_queue.qsize() +
                    'Please add more publisher threads!')
            try:
                res = self.result_queue.get(timeout=.5)
            except queue.Empty:
                continue
            features = res['features']
            topic = res['topic']
            image = res['image']
            header = res['header']
            self.result_queue.task_done()
            feature_msg = features_to_ros_msg(features, header)
            self.feature_publishers[topic].publish(feature_msg)

            # drawing keypoints is the most expensive operation in the publisher thread, and
            # can be slow when the system workload is high. So we skip drawing the keypoints
            # if there are many queued results to be published.
            if self.result_queue.qsize() > 2: continue
            if self.keypoint_publishers[topic].get_num_connections() > 0 or self.gui:
                draw_keypoints(image, features['keypoints'], features['scores'])
                if self.keypoint_publishers[topic].get_num_connections() > 0:
                    keypoint_msg = self.cv_bridge.cv2_to_imgmsg(image, encoding='passthrough')
                    keypoint_msg.header = header
                    self.keypoint_publishers[topic].publish(keypoint_msg)
                if self.gui:
                    cv2.imshow(topic, image)
                    cv2.waitKey(1)

    def print_stats(self, topic):
        now = rospy.Time.now()
        if self.stats[topic]['last_time'] is None:
            self.stats[topic]['last_time'] = now
        elapsed = (now - self.stats[topic]['last_time']).to_sec()
        if elapsed > self.log_interval:
            with self.stats_lock:
                received = self.stats[topic]['received']
                processed = self.stats[topic]['processed']
                self.stats[topic]['received'] = 0
                self.stats[topic]['processed'] = 0
                self.stats[topic]['last_time'] = now
            if received > 0:
                rospy.loginfo(topic + ': processed %d out of %d in past %.1f sec (%.2f FPS)' % (processed, received, elapsed, processed / elapsed))
            else:
                rospy.loginfo(topic + ': no message received')

    def process(self, msg, topic):
        if msg.encoding == '8UC1' or msg.encoding == 'mono8':
            image = self.cv_bridge.imgmsg_to_cv2(msg)
            image_gray = image
        else:
            image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        start = time.time()
        features = self.net.infer(image_gray)
        stop = time.time()
        rospy.logdebug(topic + ': %.2f (%d keypoints)' % (
                (stop - start) * 1000,
                features['keypoints'].shape[0]))
        if (features['keypoints'].shape[0] != 0):
            res = {'features': features, 'header': msg.header, 'topic': topic, 'image': image}
            self.result_queue.put(res)

def draw_keypoints(image, keypoints, scores):
    upper_score = 0.5
    lower_score = 0.1
    scale = 1 / (upper_score - lower_score)
    for p,s in zip(keypoints, scores):
        s = min(max(s - lower_score, 0) * scale, 1)
        color = (255 * (1 - s), 255 * (1 - s), 255) # BGR
        cv2.circle(image, tuple(p), 3, color, 2)

def features_to_ros_msg(features, header):
    msg = ImageFeatures()
    msg.header = header
    msg.sorted_by_score.data = False
    for kp in features['keypoints']:
        p = KeyPoint()
        p.x = kp[0]
        p.y = kp[1]
        msg.keypoints.append(p)
    msg.scores = features['scores'].flatten()
    msg.descriptors.data = features['local_descriptors'].flatten()
    shape = features['local_descriptors'][0].shape
    msg.descriptors.layout.dim.append(MultiArrayDimension())
    msg.descriptors.layout.dim[0].label = 'keypoint'
    msg.descriptors.layout.dim[0].size = shape[0]
    msg.descriptors.layout.dim[0].stride = shape[0] * shape[1]
    msg.descriptors.layout.dim.append(MultiArrayDimension())
    msg.descriptors.layout.dim[1].label = 'descriptor'
    msg.descriptors.layout.dim[1].size = shape[1]
    msg.descriptors.layout.dim[1].stride = shape[1]
    msg.global_descriptor = features['global_descriptor'][0]
    return msg

if __name__ == "__main__":
    main()