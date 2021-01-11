#!/usr/bin/env python3
import threading
import logging as log
import tensorflow as tf
import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork
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
    'nms_iterations': 0,
    'nms_radius': 4,
}

class InferReqWrap:
    def __init__(self, request, id, num_iter):
        self.id = id
        self.request = request
        self.num_iter = num_iter
        self.cur_iter = 0
        self.cv = threading.Condition()
        self.request.set_completion_callback(self.callback, self.id)

    def callback(self, statusCode, userdata):
        if (userdata != self.id):
            log.error("Request ID {} does not correspond to user data {}".format(self.id, userdata))
        elif statusCode != 0:
            log.error("Request {} failed with status code {}".format(self.id, statusCode))
        self.cur_iter += 1
        log.info("Completed {} Async request execution".format(self.cur_iter))
        if self.cur_iter < self.num_iter:
            # here a user can read output containing inference results and put new input
            # to repeat async request again
            self.request.async_infer(self.input)
        else:
            # continue sample execution after last Asynchronous inference request execution
            self.cv.acquire()
            self.cv.notify()
            self.cv.release()

    def execute(self, mode, input_data):
        if (mode == "async"):
            log.info("Start inference ({} Asynchronous executions)".format(self.num_iter))
            self.input = input_data
            # Start async request for the first time. Wait all repetitions of the async request
            self.request.async_infer(input_data)
            self.cv.acquire()
            self.cv.wait()
            self.cv.release()
        elif (mode == "sync"):
            log.info("Start inference ({} Synchronous executions)".format(self.num_iter))
            for self.cur_iter in range(self.num_iter):
                # here we start inference synchronously and wait for
                # last inference request execution
                self.request.infer(input_data)
                log.info("Completed {} Sync request execution".format(self.cur_iter + 1))
        else:
            log.error("wrong inference mode is chosen. Please use \"sync\" or \"async\" mode")
            sys.exit(1)



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
        self.cur_request_id = 0
        self.next_request_id = 2
       #set num_requests 推理数 ,device _name
        self.exec_net = self.ie.load_network(network=self.net, num_requests=4, device_name="CPU")


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
        return np.array(keypoints)[..., ::-1], np.array(scores)

    def select_keypoints_threshold(self, scores, keypoint_threshold, scale):
        keypoints = tf.where(tf.greater_equal(scores[0], self.config['keypoint_threshold'])).numpy()
        keypoints = np.array(keypoints)
        scores = np.array([scores[0, i[0], i[1]] for i in keypoints])
        return keypoints[..., ::-1], scores

    def infer(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = [image.shape[1] / self.input_size[0], image.shape[0] / self.input_size[1]]
        image_scaled = cv2.resize(image, self.input_size)[:,:,None]
        image_scaled = image_scaled.transpose((2, 0, 1))


         #set 4 requests 开始推理的 start_async 函数
        self.request_id = self.next_request_id       

        self.exec_net.start_async(request_id=self.request_id, inputs={self.input_blob: np.expand_dims(image_scaled, axis=0)})
        
        self.exec_net.requests[self.cur_request_id].wait(-1)

        res = self.exec_net.requests[self.cur_request_id].outputs


        features = {}
        # 1. Keypoints
        scores = self.find_first_available(res, [
            'pred/simple_nms/Select',
            'pred/local_head/detector/Squeeze'])

        if self.config['keypoint_number'] == 0 and self.config['nms_iterations'] == 0:
            keypoints, features['scores'] = self.select_keypoints_threshold(scores,
                    self.config['keypoint_threshold'], scale)
        else:
            keypoints, features['scores'] = self.select_keypoints(scores,
                    self.config['keypoint_number'], self.config['keypoint_threshold'],
                    self.config['nms_iterations'], self.config['nms_radius'])
        # scaling back
        features['keypoints'] = np.array([[int(i[0] * scale[0]), int(i[1] * scale[1])] for i in keypoints])

        # 2. Local descriptors
        if len(features['keypoints']) > 0:
            local = self.find_first_available(res, [
                'pred/local_head/descriptor/Conv_1/BiasAdd/Normalize',
                'pred/local_head/descriptor/l2_normalize'])
            local = np.transpose(local, (0,2,3,1))
            features['local_descriptors'] = \
                    tf.nn.l2_normalize(
                        tf.contrib.resampler.resampler(
                            local,
                            tf.to_float(self.scaling_desc)[::-1]*tf.to_float(keypoints[None])),
                        -1).numpy()
        else:
            features['local_descriptors'] = np.array([[]])

        # 3. Global descriptor
        features['global_descriptor'] = self.find_first_available(res, [
            'pred/global_head/l2_normalize_1',
            'pred/global_head/dimensionality_reduction/BiasAdd/Normalize'])
        
        #4 requests 并发：
        self.cur_request_id = self.cur_request_id + 1
        self.next_request_id = self.next_request_id + 1
        if self.cur_request_id ==4:
            self.cur_request_id = 0
        if self.next_request_id==4:
            self.next_request_id = 0
        return features

    @staticmethod
    def find_first_available(dic, keys):
        for key in keys:
            if key in dic: return dic[key]
        print('Could not find any of these keys:%s\nAvailable keys are:%s' % (
                ''.join(['\n\t' + key for key in keys]),
                ''.join(['\n\t' + key for key in dic.keys()])))
        raise KeyError('Given keys are not available. See the log above.')
