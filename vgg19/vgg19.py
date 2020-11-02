import scipy.io
import cv2
import numpy as np
import tensorflow as tf

layers_name = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )

def conv(inputs, w, b):
    w = tf.constant(w)
    b = tf.constant(b)
    return tf.nn.conv2d(inputs, w, [1, 1, 1, 1], "SAME") + b

def Network_vgg(inputs):
    vgg_para = scipy.io.loadmat("vgg19/vgg.mat")
    layers = vgg_para["layers"]
    feature_bank = {}
    with tf.variable_scope("vgg"):
        for i in range(len(layers_name)):
            if layers[0, i][0, 0]["type"] == "conv":
                w = layers[0, i][0, 0]["weights"][0, 0]
                b = layers[0, i][0, 0]["weights"][0, 1]
                with tf.variable_scope(str(i)):
                    inputs = conv(inputs, w, b)
            elif layers[0, i][0, 0]["type"] == "relu":
                inputs = tf.nn.relu(inputs)
                feature_bank[layers[0, i][0, 0]["name"][0]] = inputs
            else:
                inputs = tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
    return feature_bank