from __future__ import print_function
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import scipy.io
import scipy.misc
from cluster_DLL import *

class netlayer:
    def __init__(self, type, pSize = 5, kernels = np.array([]), bias = np.array([])):
        self.type = type # either "cos" or "maxpool"
        self.bias = bias
        self.kernels = []
        self.pSize = pSize
        self.channels = 0
        if kernels.shape[0] > 0:
            self.channels = int(kernels[0].shape[0]/(pSize * pSize))
            for i in range (kernels.shape[0]):
                w = kernels[i, :].reshape([pSize, pSize, self.channels, 1])
                self.kernels.append(w)




def cosine_Convolution (input, w, padding = "VALID", stride = 1):
    # input: input image, [batch, hight, widthm channel], tf tensor
    # w: [hight, width, channel, output channels], tf tensor

    one_mask = tf.ones(w.shape, tf.float32)
    C0 = tf.nn.conv2d(input, w, strides = [1,stride,stride,1], padding = padding)

    I1 = tf.sqrt(tf.nn.conv2d(tf.square(input)  + 1e-5, one_mask, strides = [1,stride,stride,1],
                              padding = padding))
    R = tf.divide(C0, I1)
    return R

def CosCovLayer(I, weights, padding = "VALID", stride = 1):
    # I: input image/covmaps
    # weight: a list of np arrays.
    #     each element is a kernel with size [hight, width, channel, 1]
    w = weights[0]
    for i in range(1,len(weights)):
        w = np.append(w, weights[i], axis = 3)

    filter = tf.Variable ( initial_value=w )
    filter_norm = 0.05*tf.nn.l2_normalize(filter, axis = (0, 1, 2))
    Cov_map= cosine_Convolution(I, filter, padding = padding, stride = stride)
    # init = tf.truncated_normal((1,1,1, len(weights)), stddev=0.001, dtype=tf.float32)
    # b = tf.Variable(initial_value= 0.02*np.ones([1,1,1, len(weights)], dtype = np.float32))
    b = tf.Variable(tf.zeros([1, 1, 1, len(weights)], tf.float32))
    return Cov_map+b, [filter, b ]

#
# def resblock(input, pSize, nkernels,  resinitializer, name = None):
#
#     # input_BN = tf.contrib.layers.batch_norm(input)
#     resconv1 = tf.layers.conv2d(input, nkernels, (pSize, pSize), activation=tf.nn.leaky_relu, name=name+ '_1',
#                                   kernel_initializer=resinitializer, padding='SAME')
#     # resconv1BN = tf.contrib.layers.batch_norm(resconv1)
#     resconv2 = tf.layers.conv2d(resconv1, nkernels, (3, 3), activation=None, name=name+'_2',
#                                   kernel_initializer=resinitializer, padding='SAME')
#     return input + resconv2
#
# def resblock_bottleneck(input, pSize, nkernels,  resinitializer, name = None):
#     resconv1 = tf.layers.conv2d(input, nkernels/2, (1, 1), activation=tf.nn.leaky_relu, name=name + '_1',
#                                 kernel_initializer=resinitializer, padding='SAME')
#     resconv2 = tf.layers.conv2d(resconv1, nkernels, (3, 3), activation=tf.nn.leaky_relu, name=name + '_2',
#                                 kernel_initializer=resinitializer, padding='SAME')
#     resconv3 = tf.layers.conv2d(resconv2, nkernels, (1, 1), activation=None, name=name + '_3',
#                                 kernel_initializer=resinitializer, padding='SAME')
#     return input + resconv3


def resblock(input, pSize, nkernels,  resinitializer, training, stride = (1,1), name = None, BN = False, prj = False):
    if BN:

        resconv1 = tf.layers.conv2d(input, nkernels, (pSize, pSize), activation=None, name=name+ '_1',
                                  strides=stride, kernel_initializer=resinitializer, padding='SAME')
        resconv1_BN =  tf.nn.leaky_relu(tf.layers.batch_normalization(resconv1, training = training))

        resconv2 = tf.layers.conv2d(resconv1_BN,
                                    nkernels, (3, 3), activation=None,
                                    name=name + '_2',
                                    kernel_initializer=resinitializer, padding='SAME')
        resconv2_BN = tf.nn.leaky_relu(tf.layers.batch_normalization(resconv2, training=training))
        if prj:
            input_prj = tf.layers.conv2d(input, nkernels, (1, 1), activation=tf.nn.leaky_relu, name=name+ '_prj',
                                         strides=stride, kernel_initializer=resinitializer, padding='SAME')
            return input_prj + resconv2_BN
        else:
            return input + resconv2_BN
    else:
        resconv1 = tf.layers.conv2d(input, nkernels, (pSize, pSize),
                                    activation=tf.nn.leaky_relu, name=name + '_1',
                                    kernel_initializer=resinitializer, padding='SAME')
        resconv2 = tf.layers.conv2d(resconv1, nkernels, (3, 3), activation=None,
                                    name=name + '_2',
                                    kernel_initializer=resinitializer, padding='SAME')
        if prj:
            input_prj = tf.layers.conv2d(input, nkernels, (pSize, pSize), activation=tf.nn.leaky_relu,
                                         name=name + '_prj',
                                         kernel_initializer=resinitializer, padding='SAME')
            return input_prj + resconv2
        else:
            return input + resconv2
