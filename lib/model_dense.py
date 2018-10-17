from __future__ import absolute_import, division, print_function

import collections
import math
import os

import numpy as np
import scipy.misc as sic
import tensorflow as tf
import tensorflow.contrib.slim as slim

from lib.ops import (batchnorm, conv2, denselayer, lrelu, pixelShuffler,
                     prelu_tf)


# The dense layer
def denseConvlayer(layer_inputs, bottleneck_scale, growth_rate, is_training):
    # Build the bottleneck operation
    net = layer_inputs
    net_temp = tf.identity(net)
    # net = batchnorm(net, is_training)
    net = prelu_tf(net, name='Prelu_1')
    net = conv2(net, kernel=1, output_channel=bottleneck_scale*growth_rate, stride=1, use_bias=False, scope='conv1x1')
    # net = batchnorm(net, is_training)
    net = prelu_tf(net, name='Prelu_2')
    net = conv2(net, kernel=3, output_channel=growth_rate, stride=1, use_bias=False, scope='conv3x3')

    # Concatenate the processed feature to the feature
    net = tf.concat([net_temp, net], axis=3)

    return net


# The transition layer
def transitionLayer(layer_inputs, output_channel, is_training):
    net = layer_inputs
    # net = batchnorm(net, is_training)
    net = prelu_tf(net)
    net = conv2(net, 1, output_channel, stride=1, use_bias=False, scope='conv1x1')

    return net


# The dense block
def denseBlock(block_inputs, num_layers, bottleneck_scale, growth_rate, is_training=None):
    # Build each layer consecutively
    net = block_inputs
    for i in range(num_layers):
        with tf.variable_scope('dense_conv_layer%d'%(i+1)):
            net = denseConvlayer(net, bottleneck_scale, growth_rate, is_training)

    return net


# Here we define the dense block version generator
def generatorDense(gen_inputs, gen_output_channels, reuse=False, is_training=None):
    # The main netowrk
    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input stage
        with tf.variable_scope('input_stage'):
            net = conv2(gen_inputs, 9, 64, 1, scope='conv')
            net = prelu_tf(net)

        # The dense block part
        # Define the denseblock configuration
        layer_per_block = 16
        bottleneck_scale = 4
        growth_rate = 12
        transition_output_channel = 128
        with tf.variable_scope('denseBlock_1'):
            net = denseBlock(net, layer_per_block, bottleneck_scale, growth_rate, is_training)

        with tf.variable_scope('transition_layer_1'):
            net = transitionLayer(net, transition_output_channel, is_training)

        with tf.variable_scope('subpixelconv_stage1'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('subpixelconv_stage2'):
            net = conv2(net, 3, 256, 1, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('output_stage'):
            net = conv2(net, 9, gen_output_channels, 1, scope='conv')

        return net


# Definition of dense block version the discriminator
def discriminatorDense(dis_inputs, is_training=True):
    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1')
            # net = batchnorm(net, is_training)
            net = lrelu(net, 0.2)
        return net

    with tf.variable_scope('discriminator_unit'):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv2(dis_inputs, 3, 64, 1, scope='conv')
            net = lrelu(net, 0.2)

        # The discriminator block part
        # block 1
        net = discriminator_block(net, 64, 3, 2, 'disblock_1')

        # block 2
        net = discriminator_block(net, 64, 3, 2, 'disblock_2')

        # block 3
        net = discriminator_block(net, 64, 3, 1, 'disblock_3')

        # The dense block part
        # Define the denseblock configuration
        layer_per_block = 8
        bottleneck_scale = 4
        growth_rate = 12
        transition_output_channel = 128
        with tf.variable_scope('denseBlock_1'):
            net = denseBlock(net, layer_per_block, bottleneck_scale, growth_rate, is_training)

        with tf.variable_scope('transition_layer_1'):
            net = transitionLayer(net, transition_output_channel, is_training)

        # The dense layer 1
        with tf.variable_scope('dense_layer_1'):
            net = slim.flatten(net)
            net = denselayer(net, 1024)
            net = lrelu(net, 0.2)

        # The dense layer 2
        with tf.variable_scope('dense_layer_2'):
            net = denselayer(net, 1)
            net = tf.nn.sigmoid(net)
    return net
