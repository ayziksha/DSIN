from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.contrib.layers.python.layers import initializers

""" This function gets a concatenated (x_dec + y_si) - 6 channel input and returns x_final (3 chanel)"""

def lrelu(x):
    return tf.maximum(x*0.2,x)


def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(shape[2]):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer


def nm(x):  # changed to None
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x)


def siNet(input):

    net=slim.conv2d(input,32,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv1', data_format='NCHW')
    net=slim.conv2d(net,32,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv2', data_format='NCHW')
    net=slim.conv2d(net,32,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv3', data_format='NCHW')
    net=slim.conv2d(net,32,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv4', data_format='NCHW')
    net=slim.conv2d(net,32,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv5', data_format='NCHW')
    net=slim.conv2d(net,32,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv6', data_format='NCHW')
    net=slim.conv2d(net,32,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv7', data_format='NCHW')
    net=slim.conv2d(net,32,[3,3],rate=128,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv8', data_format='NCHW')
    net=slim.conv2d(net,32,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=None,weights_initializer=identity_initializer(),scope='g_conv9', data_format='NCHW')
    net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_conv_last', data_format='NCHW')
    return net