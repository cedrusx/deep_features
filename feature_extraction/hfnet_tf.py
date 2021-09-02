#!/usr/bin/env python3

# Copyright (C) <2020-2021> Intel Corporation
# SPDX-License-Identifier: MIT

import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
tf.contrib.resampler

default_config = {
    'model_path': 'models/hfnet_tf',
    'keypoint_number': 500,
    'keypoint_threshold': 0.002,
    'nms_iterations': 1,
    'nms_radius': 1,
}


class FeatureNet:
    def __init__(self, config=default_config):
        self.graph = tf.Graph()
        self.graph.as_default()
        self.sess = tf.Session(graph=self.graph)
        tf.saved_model.loader.load(
                self.sess,
                [tag_constants.SERVING],
                config['model_path'])
        self.net_image_in = self.graph.get_tensor_by_name('image:0')
        self.net_scores = self.graph.get_tensor_by_name('scores:0')
        self.net_logits = self.graph.get_tensor_by_name('logits:0')
        self.net_local_desc = self.graph.get_tensor_by_name('local_descriptors:0')
        self.net_global_decs = self.graph.get_tensor_by_name('global_descriptor:0')
        self.keypoints, self.scores = self.select_keypoints(
                self.net_scores, config['keypoint_number'], config['keypoint_threshold'],
                config['nms_iterations'], config['nms_radius'])
        # inverse ratio for upsampling (should be approx. 1/8)
        self.scaling_op = ((tf.cast(tf.shape(self.net_local_desc)[1:3], tf.float32) - 1.)
            / (tf.cast(tf.shape(self.net_image_in)[1:3], tf.float32) - 1.))
        # bicubic interpolation (upsample X8 to the image size) and L2-normalization
        self.local_descriptors = \
            tf.nn.l2_normalize(
                tf.contrib.resampler.resampler(
                    self.net_local_desc,
                    self.scaling_op[::-1] * tf.to_float(self.keypoints)[None]),
                -1)


    def simple_nms(self, scores, iterations, radius):
        """Performs non maximum suppression (NMS) on the heatmap using max-pooling.
        This method does not suppress contiguous points that have the same score.
        It is an approximate of the standard NMS and uses iterative propagation.
        Arguments:
            scores: the score heatmap, with shape `[B, H, W]`.
            size: an interger scalar, the radius of the NMS window.
        """
        if iterations < 1: return scores
        with self.graph.as_default():
            with tf.name_scope('simple_nms'):
                radius = tf.constant(radius, name='radius')
                size = radius*2 + 1

                max_pool = lambda x: gen_nn_ops.max_pool_v2(  # supports dynamic ksize
                        x[..., None], ksize=[1, size, size, 1],
                        strides=[1, 1, 1, 1], padding='SAME')[..., 0]
                zeros = tf.zeros_like(scores)
                max_mask = tf.equal(scores, max_pool(scores))
                for _ in range(iterations-1):
                    supp_mask = tf.cast(max_pool(tf.to_float(max_mask)), tf.bool)
                    supp_scores = tf.where(supp_mask, zeros, scores)
                    new_max_mask = tf.equal(supp_scores, max_pool(supp_scores))
                    max_mask = max_mask | (new_max_mask & tf.logical_not(supp_mask))
                return tf.where(max_mask, scores, zeros)


    def select_keypoints(self, scores, keypoint_number, keypoint_threshold, nms_iterations, nms_radius):
        with self.graph.as_default():
            scores = self.simple_nms(scores, nms_iterations, nms_radius)
            with tf.name_scope('keypoint_extraction'):
                keypoints = tf.where(tf.greater_equal(
                    scores[0], keypoint_threshold))
                scores = tf.gather_nd(scores[0], keypoints)
            with tf.name_scope('top_k_keypoints'):
                k = tf.constant(keypoint_number, name='k')
                k = tf.minimum(tf.shape(scores)[0], k)
                scores, indices = tf.nn.top_k(scores, k)
                keypoints = tf.to_int32(tf.gather(
                    tf.to_float(keypoints), indices))
            keypoints = keypoints[..., ::-1]  # x-y convention
            return keypoints, scores


    def infer(self, image):
        if len(image.shape) == 2: # grayscale
            image_in = image[None,:,:,None]
        else:
            image_in = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[None,:,:,None]
        results = self.sess.run(
                [
                    self.keypoints,         # (num_keypoints, 2) int64
                    self.scores,            # (num_keypoints,) float32
                    self.local_descriptors, # (1, num_keypoints, 256) float32
                    self.net_global_decs,   # (1, 4096) float32
                ],
                feed_dict = {self.net_image_in: image_in})

        features = {}
        features['keypoints'] = results[0]
        features['scores'] = results[1]
        features['local_descriptors'] = results[2]
        features['global_descriptor'] = results[3]
        return features
