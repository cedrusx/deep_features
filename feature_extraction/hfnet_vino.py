#!/usr/bin/env python3

import tensorflow as tf
import cv2
import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin,IECore
import os
from tensorflow.python.ops import gen_nn_ops
tf.enable_eager_execution()

default_config = {
    'cpu_extension': "/opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so",
    'model_path': 'models/hfnet_vino',
    'model_file': "hfnet.xml",
    'weights_file': "hfnet.bin",
    'keypoint_number': 500,
    'keypoint_threshold': 0.002,
    'nms_iterations': 1,
    'nms_radius': 1,
}

class FeatureNet:
    def __init__(self, config=default_config):
        self.config = config
        self.ie = IECore()
        if os.path.exists(config['cpu_extension']):
            self.ie.add_extension(config['cpu_extension'], 'CPU')
        else:
            print('CPU extension file does not exist: %s' % config['cpu_extension'])
        model = os.path.join(config['model_path'], config['model_file'])
        weights = os.path.join(config['model_path'], config['weights_file'])
        self.net = IENetwork(model=model, weights=weights)
        # Input size is specified by the OpenVINO model
        input_shape = self.net.inputs['image'].shape
        self.input_size = (input_shape[3], input_shape[2])
        self.scaling_desc = (np.array(self.input_size) / 8 - 1.) / (np.array(self.input_size) - 1.)
        print('OpenVINO model input size: (%d, %d)' % (self.input_size[0], self.input_size[1]))
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        self.net.batch_size = 1
        self.exec_net = self.ie.load_network(network=self.net, device_name="CPU")

    def simple_nms(self, scores, iterations, radius):
        """Performs non maximum suppression (NMS) on the heatmap using max-pooling.
        This method does not suppress contiguous points that have the same score.
        It is an approximate of the standard NMS and uses iterative propagation.
        Arguments:
            scores: the score heatmap, with shape `[B, H, W]`.
            size: an interger scalar, the radius of the NMS window.
        """
        if iterations < 1: return scores
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
        scores = self.simple_nms(scores, nms_iterations, nms_radius)
        keypoints = tf.where(tf.greater_equal(
            scores[0], keypoint_threshold))
        scores = tf.gather_nd(scores[0], keypoints)
        k = tf.constant(keypoint_number, name='k')
        k = tf.minimum(tf.shape(scores)[0], k)
        scores, indices = tf.nn.top_k(scores, k)
        keypoints = tf.to_int32(tf.gather(
            tf.to_float(keypoints), indices))
        return np.array(keypoints), np.array(scores)

    def select_keypoints_threshold(self, scores, keypoint_threshold, scale):
        keypoints = tf.where(tf.greater_equal(scores[0], self.config['keypoint_threshold'])).numpy()
        keypoints = np.array(keypoints)
        scores = np.array([scores[0, i[0], i[1]] for i in keypoints])
        return keypoints, scores

    def infer(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = [image.shape[1] / self.input_size[0], image.shape[0] / self.input_size[1]]
        image_scaled = cv2.resize(image, self.input_size)[:,:,None]
        image_scaled = image_scaled.transpose((2, 0, 1))
        res = self.exec_net.infer(inputs={self.input_blob: np.expand_dims(image_scaled, axis=0)})

        features = {}
        scores = res['pred/local_head/detector/Squeeze']
        if self.config['keypoint_number'] == 0 and self.config['nms_iterations'] == 0:
            keypoints, features['scores'] = self.select_keypoints_threshold(scores,
                    self.config['keypoint_threshold'], scale)
        else:
            keypoints, features['scores'] = self.select_keypoints(scores,
                    self.config['keypoint_number'], self.config['keypoint_threshold'],
                    self.config['nms_iterations'], self.config['nms_radius'])
        # scaling back and x-y conversion
        features['keypoints'] = np.array([[int(i[1] * scale[0]), int(i[0] * scale[1])] for i in keypoints])

        local = np.transpose(res['pred/local_head/descriptor/Conv_1/BiasAdd/Normalize'],(0,2,3,1))
        if len(features['keypoints']) > 0:
            features['local_descriptors'] = \
                    tf.nn.l2_normalize(
                        tf.contrib.resampler.resampler(
                            local,
                            tf.to_float(self.scaling_desc)[::-1]*tf.to_float(features['keypoints'][None])),
                        -1).numpy()
        else:
            features['local_descriptors'] = np.array([[]])

        features['global_descriptor'] = res['pred/global_head/dimensionality_reduction/BiasAdd/Normalize']

        return features
