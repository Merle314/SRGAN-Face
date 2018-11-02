from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocessLR(image):
    with tf.name_scope("preprocessLR"):
        return tf.identity(image)


def deprocessLR(image):
    with tf.name_scope("deprocessLR"):
        return tf.identity(image)


# Define the convolution building block
def conv2(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv'):
    # kernel: An integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NHWC',
                            activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=None)


def conv2_NCHW(batch_input, kernel=3, output_channel=64, stride=1, use_bias=True, scope='conv_NCHW'):
    # Use NCWH to speed up the inference
    # kernel: list of 2 integer specifying the width and height of the 2D convolution window
    with tf.variable_scope(scope):
        if use_bias:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NCWH',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer())
        else:
            return slim.conv2d(batch_input, output_channel, [kernel, kernel], stride, 'SAME', data_format='NCWH',
                               activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),
                               biases_initializer=None)


# Define our tensorflow version PRelu
def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5

    return pos + neg


# Define our Lrelu
def lrelu(inputs, alpha):
    return tf.nn.leaky_relu(inputs, alpha=alpha)


def batchnorm(inputs, is_training):
    return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.GraphKeys.UPDATE_OPS,
                        scale=False, fused=True, is_training=is_training)


# Our dense layer
def denselayer(inputs, output_size):
    output = tf.layers.dense(inputs, output_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())
    return output


# The implementation of PixelShuffler
def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output


def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)


# The random flip operation used for loading examples
def random_flip(input, decision):
    f1 = tf.identity(input)
    f2 = tf.image.flip_left_right(input)
    output = tf.cond(tf.less(decision, 0.5), lambda: f2, lambda: f1)

    return output


# The operation used to print out the configuration
def print_configuration_op(FLAGS):
    print('[Configurations]:')
    for name, value in FLAGS.__flags.items():
        if type(value) == float:
            print('\t%s: %f'%(name, value))
        elif type(value) == int:
            print('\t%s: %d'%(name, value))
        elif type(value) == str:
            print('\t%s: %s'%(name, value))
        elif type(value) == bool:
            print('\t%s: %s'%(name, value))
        else:
            print('\t%s: %s' % (name, value))

    print('End of configuration')


def compute_psnr(ref, target):
    ref = tf.cast(ref, tf.float32)
    target = tf.cast(target, tf.float32)
    diff = target - ref
    sqr = tf.multiply(diff, diff)
    err = tf.reduce_sum(sqr)
    v = tf.shape(diff)[0] * tf.shape(diff)[1] * tf.shape(diff)[2] * tf.shape(diff)[3]
    mse = err / tf.cast(v, tf.float32)
    psnr = 10. * (tf.log(255. * 255. / mse) / tf.log(10.))
    return psnr

# Define the convolution block before the sub_pixel layer
def subpixel_pre(batch_input, input_channel=64, output_channel=256, scope='conv'):
    output_channel=int(output_channel/4)
    with tf.variable_scope(scope):
        kernel_1 = tf.get_variable('kernel_1', shape=[3, 3, input_channel, output_channel],
                                  initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        kernel_2 = tf.get_variable('kernel_2', initializer=kernel_1.initialized_value(), dtype=tf.float32)
        kernel_3 = tf.get_variable('kernel_3', initializer=kernel_1.initialized_value(), dtype=tf.float32)
        kernel_4 = tf.get_variable('kernel_4', initializer=kernel_1.initialized_value(), dtype=tf.float32)
        kernel_1_split = tf.split(kernel_1, output_channel, axis=3)
        kernel_2_split = tf.split(kernel_2, output_channel, axis=3)
        kernel_3_split = tf.split(kernel_3, output_channel, axis=3)
        kernel_4_split = tf.split(kernel_4, output_channel, axis=3)
        kernel_concat = []
        for i in range(output_channel):
            kernel_concat = kernel_concat + [kernel_1_split[i], kernel_2_split[i], kernel_3_split[i], kernel_4_split[i]]
        kernel = tf.concat(kernel_concat, axis=3)
        return tf.nn.conv2d(batch_input, kernel, strides=[1, 1, 1, 1], padding='SAME')

def kernel_norm(kernel):
    kernel_mean = tf.reduce_mean(kernel, axis=[0, 1, 3], keep_dims=True)
    kernel_mean_per = tf.reduce_mean(kernel, axis=[0, 1], keep_dims=True)
    return kernel/(kernel_mean_per+0.1)

def kernel_constrain(kernel):
    kernel_mean = tf.reduce_mean(kernel, axis=[0, 1, 3], keep_dims=True)
    kernel_mean_per = tf.reduce_mean(kernel, axis=[0, 1], keep_dims=True)
    tf.add_to_collection('kernel_constrain', tf.reduce_mean(tf.square(1.0-kernel_mean_per)))

def perchannel_conv(inputs, kernel, input_channel):
    input_split = tf.split(inputs, input_channel, axis=3)
    return tf.concat([tf.nn.conv2d(x, kernel, strides=[1, 2, 2, 1], padding='VALID') for x in input_split], axis=3)

# Define the convolution block before the sub_pixel layer
# [1, 2
#  3, 4]
# position is above
def relate_conv(batch_input, input_channel=64, output_channel=64, scope='relate_conv'):
    with tf.variable_scope(scope):
        kernel = tf.get_variable('kernel', shape=[output_channel, 5, 5, input_channel],
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        relate_kernel = tf.ones([2, 2, 1, 1])
        kernel_1 = tf.pad(kernel, [[0, 0], [0, 1], [0, 1], [0, 0]], "CONSTANT")
        kernel_2 = tf.pad(kernel, [[0, 0], [0, 1], [1, 0], [0, 0]], "CONSTANT")
        kernel_3 = tf.pad(kernel, [[0, 0], [1, 0], [0, 1], [0, 0]], "CONSTANT")
        kernel_4 = tf.pad(kernel, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")

        kernel_1 = tf.transpose(perchannel_conv(kernel_1, relate_kernel, input_channel), [1, 2, 3, 0])
        kernel_2 = tf.transpose(perchannel_conv(kernel_2, relate_kernel, input_channel), [1, 2, 3, 0])
        kernel_3 = tf.transpose(perchannel_conv(kernel_3, relate_kernel, input_channel), [1, 2, 3, 0])
        kernel_4 = tf.transpose(perchannel_conv(kernel_4, relate_kernel, input_channel), [1, 2, 3, 0])

        kernel_1_split = tf.split(kernel_1, output_channel, axis=3)
        kernel_2_split = tf.split(kernel_2, output_channel, axis=3)
        kernel_3_split = tf.split(kernel_3, output_channel, axis=3)
        kernel_4_split = tf.split(kernel_4, output_channel, axis=3)

        kernel_concat = []
        for i in range(output_channel):
            # kernel_nor = kernel_norm(tf.concat([kernel_1_split[i], kernel_2_split[i], kernel_3_split[i], kernel_4_split[i]], axis=3))
            kernel_concat = kernel_concat + [kernel_1_split[i], kernel_2_split[i], kernel_3_split[i], kernel_4_split[i]]
            # kernel_concat.append(kernel_nor)
        kernel = tf.concat(kernel_concat, axis=3)
        with tf.control_dependencies([kernel]):
            net = tf.nn.conv2d(batch_input, kernel, strides=[1, 1, 1, 1], padding='SAME')
        return net


def resize_conv(batch_input, input_channel=64, output_channel=64, scope='resize_conv'):
    original_size = tf.shape(batch_input)
    new_size = [original_size[1]*2, original_size[2]*2]
    batch_input = tf.image.resize_nearest_neighbor(batch_input, new_size)
    return conv2(batch_input, kernel=3, output_channel=4, use_bias=False)


def rot90(tensor, k=1, axes=[0, 1], name=None):
    axes = tuple(axes)
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")
    tenor_shape = (tensor.get_shape().as_list())
    dim = len(tenor_shape)
    if axes[0] == axes[1] or np.absolute(axes[0] - axes[1]) == dim:
        raise ValueError("Axes must be different.")
    if (axes[0] >= dim or axes[0] < -dim or axes[1] >= dim or axes[1] < -dim):
        raise ValueError("Axes={} out of range for tensor of ndim={}.".format(
            axes, dim))
    k %= 4
    if k == 0:
        return tensor
    if k == 2:
        img180 = tf.reverse(
            tf.reverse(tensor, axis=[axes[0]]), axis=[axes[1]], name=name)
        return img180

    axes_list = np.arange(0, dim)
    (axes_list[axes[0]], axes_list[axes[1]]) = (axes_list[axes[1]],
                                                axes_list[axes[0]])

    print(axes_list)
    if k == 1:
        img90 = tf.transpose(
            tf.reverse(tensor, axis=[axes[1]]), perm=axes_list, name=name)
        return img90
    if k == 3:
        img270 = tf.reverse(
            tf.transpose(tensor, perm=axes_list), axis=[axes[1]], name=name)
        return img270

def interpolation_kernel(shape, name='interpolation_kernel'):
    inter_kernel = tf.get_variable(name, shape=[3, 3, 4],
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    # # inter_kernel = tf.abs(inter_kernel)
    # # inter_kernel = tf.constant([[[0], [0], [0]], [[0], [1], [0]], [[0], [0], [0]]], dtype=tf.float32)
    # inter_kernel1 = rot90(inter_kernel, axes=[0, 1], k=2)
    # inter_kernel2 = rot90(inter_kernel, axes=[0, 1], k=1)
    # inter_kernel3 = rot90(inter_kernel, axes=[0, 1], k=3)
    # inter_kernel4 = tf.identity(inter_kernel)
    # inter_concat = tf.concat([inter_kernel1, inter_kernel2, inter_kernel3, inter_kernel4], axis=2)
    inter_kernel = tf.reshape(inter_kernel, [3, 3, 2, 2])
    inter_kernel = tf.transpose(inter_kernel, [0, 2, 1, 3])
    return tf.reshape(inter_kernel, shape)

# Define the convolution block before the sub_pixel layer
# [1, 2
#  3, 4]
# position is above
def interpolation_conv(batch_input, input_channel=64, output_channel=64, scope='relate_conv'):
    with tf.variable_scope(scope):
        kernel = tf.get_variable('kernel', shape=[output_channel, 6, 6, input_channel],
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        # inter_kernel = interpolation_kernel([6, 6, 1, 1], name='interpolation_kernel')
        inter_kernel = tf.get_variable('interpolation_kernel', shape=[6, 6, 1, 1],
                                 initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        # kernel_1 = tf.pad(kernel, [[0, 0], [2, 3], [2, 3], [0, 0]], "CONSTANT")
        # kernel_2 = tf.pad(kernel, [[0, 0], [2, 3], [3, 2], [0, 0]], "CONSTANT")
        # kernel_3 = tf.pad(kernel, [[0, 0], [3, 2], [2, 3], [0, 0]], "CONSTANT")
        # kernel_4 = tf.pad(kernel, [[0, 0], [3, 2], [3, 2], [0, 0]], "CONSTANT")
        kernel_1 = tf.pad(kernel, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
        kernel_2 = tf.pad(kernel, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
        kernel_3 = tf.pad(kernel, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
        kernel_4 = tf.pad(kernel, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")

        kernel_1 = tf.transpose(perchannel_conv(kernel_1, inter_kernel, input_channel), [1, 2, 3, 0])
        kernel_2 = tf.transpose(perchannel_conv(kernel_2, inter_kernel, input_channel), [1, 2, 3, 0])
        kernel_3 = tf.transpose(perchannel_conv(kernel_3, inter_kernel, input_channel), [1, 2, 3, 0])
        kernel_4 = tf.transpose(perchannel_conv(kernel_4, inter_kernel, input_channel), [1, 2, 3, 0])

        kernel_1_split = tf.split(kernel_1, output_channel, axis=3)
        kernel_2_split = tf.split(kernel_2, output_channel, axis=3)
        kernel_3_split = tf.split(kernel_3, output_channel, axis=3)
        kernel_4_split = tf.split(kernel_4, output_channel, axis=3)

        kernel_concat = []
        for i in range(output_channel):
            kernel_concat = kernel_concat + [kernel_1_split[i], kernel_2_split[i], kernel_3_split[i], kernel_4_split[i]]
        kernel = tf.concat(kernel_concat, axis=3)
        with tf.control_dependencies([kernel]):
            net = tf.nn.conv2d(batch_input, kernel, strides=[1, 1, 1, 1], padding='SAME')
        return net




def get_dist_matrix():
    # distance matrix for blur process
    kernel_radius = 3
    kernel_size = int(np.ceil(6*kernel_radius))
    kernel_size = kernel_size + 1 if kernel_size%2==0 else kernel_size
    distance_matrix = np.zeros([kernel_size, kernel_size])
    center = kernel_size/2
    for i in range(kernel_size):
        for j in range(kernel_size):
            distance_matrix[i,j]= ((center-i)**2 + (center-j)**2)**.5
    distance_matrix = np.expand_dims(distance_matrix,2)
    distance_matrix = np.expand_dims(distance_matrix,3)
    return distance_matrix

def blur(image, sigma, device_id):
    channel = image.shape[2]
    dist = get_dist_matrix()
    kernel = tf.exp(-dist/(2*sigma**2))
    kernel = kernel/tf.reduce_sum(kernel)
    image = tf.expand_dims(image,0)
    image_split = tf.split(image, channel, axis=3)
    with tf.device('/gpu:%d'%device_id):
        processed = tf.concat([tf.nn.conv2d(x, kernel, [1, 1, 1, 1], 'SAME') for x in image_split], axis=3)
        # processed = tf.nn.conv2d(image, kernel, [1,1,1,1], 'SAME')
    processed = tf.squeeze(processed, axis=0)
    return processed

def random_blur(image, min_sigma, max_sigma, device_id, always=True):
    def _random_adjust(image):
        sigma = tf.random_uniform([], minval=min_sigma, maxval=max_sigma)
        return blur(image, sigma, device_id)
    if always:
        return _random_adjust(image)
    else:
        rand_value = tf.random_uniform([])
        rand_cond = tf.greater_equal(rand_value, 0.5)
        return tf.cond(rand_cond, lambda: _random_adjust(image),
                                  lambda: image)
