#!/usr/bin/env python3
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

def main():
    rospy.init_node('feature_extraction_node')
    net_name = rospy.get_param('~net', 'hfnet_tf')
    # user can set more than one input image topics, e.g. /cam1/image,/cam2/image
    topics = rospy.get_param('~topics', '/d400/color/image_raw')
    gui = rospy.get_param('~gui', True)
    if net_name == 'hfnet_vino':
        from hfnet_vino import FeatureNet, default_config
    elif net_name == 'hfnet_tf':
        from hfnet_tf import FeatureNet, default_config
    else:
        exit('Unknown net %s' % net_name)
    config = default_config
    config['keypoint_number'] = rospy.get_param('~keypoint_number', config['keypoint_number'])
    config['model_path'] = rospy.get_param('~model_path', config['model_path'])
    net = FeatureNet()
    node = Node(net, gui)
    for topic in topics.split(','):
        node.subscribe(topic)
    rospy.spin()

class Node():
    def __init__(self, net, gui):
        self.net = net
        self.gui = gui
        self.cv_bridge = CvBridge()
        self.publishers = {}
        self.subscribers = {}
        self.latest_msgs = {}
        self.lock = threading.Lock() # protect latest_msgs
        self.thread = threading.Thread(target=self.worker)
        self.thread.start()

    def subscribe(self, topic):
        output_topic = '/'.join(topic.split('/')[:-1]) + '/features'
        self.publishers[topic] = rospy.Publisher(output_topic, ImageFeatures, queue_size=1)
        with self.lock:
            self.latest_msgs[topic] = None
        callback = lambda msg: self.callback(msg, topic)
        self.subscribers[topic] = rospy.Subscriber(topic, Image, callback, queue_size=1)

    def callback(self, msg, topic):
        # keep only the lastest message
        with self.lock:
            self.latest_msgs[topic] = msg

    def worker(self):
        while not rospy.is_shutdown():
            no_new_msg = True
            # take turn to process each topic
            for topic in self.latest_msgs.keys():
                with self.lock:
                    msg = self.latest_msgs[topic]
                    self.latest_msgs[topic] = None
                if msg is None:
                    rospy.loginfo_throttle(3, topic + ': no message received')
                    continue
                self.process(msg, topic)
                no_new_msg = False
            if no_new_msg: time.sleep(0.01)

    def process(self, msg, topic):
        start_time = time.time()
        if msg.encoding == '8UC1' or msg.encoding == 'mono8':
            image_gray = self.cv_bridge.imgmsg_to_cv2(msg)
            if self.gui: image_color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        else:
            image_color = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
            image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        t2 = time.time()
        features = self.net.infer(image_gray)
        t3 = time.time()
        if (features['keypoints'].shape[0] != 0):
            feature_msg = features_to_ros_msg(features, msg)
            self.publishers[topic].publish(feature_msg)
        end_time = time.time()
        rospy.loginfo(topic + ': %.2f | %.2f ms (%d keypoints)' % (
                (end_time-start_time) * 1000,
                (t3 - t2) * 1000,
                features['keypoints'].shape[0]))
        if self.gui:
            draw_keypoints(image_color, features['keypoints'], features['scores'])
            cv2.imshow(topic, image_color)
            cv2.waitKey(1)

def draw_keypoints(image, keypoints, scores):
    upper_score = 0.5
    lower_score = 0.1
    scale = 1 / (upper_score - lower_score)
    for p,s in zip(keypoints, scores):
        s = min(max(s - lower_score, 0) * scale, 1)
        color = (255 * (1 - s), 255 * (1 - s), 255) # BGR
        cv2.circle(image, tuple(p), 3, color, 2)

def features_to_ros_msg(features, img_msg):
    msg = ImageFeatures()
    msg.header = img_msg.header
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