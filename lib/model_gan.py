from __future__ import absolute_import, division, print_function

import collections
import math
import os

import numpy as np
import scipy.misc as sic
import tensorflow as tf
import tensorflow.contrib.slim as slim

from lib.ops import (batchnorm, conv2, denselayer, lrelu, pixelShuffler,
                     prelu_tf, subpixel_pre)


# Definition of the generator
def generator(gen_inputs, gen_output_channels, reuse=False, FLAGS=None):
    # Check the flag
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    # The Bx residual blocks
    def residual_block(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, 3, output_channel, stride, use_bias=False, scope='conv_1')
            # net = batchnorm(net, FLAGS.is_training)
            net = prelu_tf(net)
            net = conv2(net, 3, output_channel, stride, use_bias=False, scope='conv_2')
            # net = batchnorm(net, FLAGS.is_training)
            net = net + inputs

        return net

    with tf.variable_scope('generator_unit', reuse=reuse):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv2(gen_inputs, 9, 64, 1, scope='conv')
            net = prelu_tf(net)

        stage1_output = net

        # The residual block parts
        for i in range(1, FLAGS.num_resblock+1 , 1):
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = conv2(net, 3, 64, 1, use_bias=False, scope='conv')
            # net = batchnorm(net, FLAGS.is_training)

        net = net + stage1_output

        with tf.variable_scope('subpixelconv_stage1'):
            # net = conv2(net, 3, 256, 1, scope='conv')
            net = subpixel_pre(net, 64, 256, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('subpixelconv_stage2'):
            # net = conv2(net, 3, 256, 1, scope='conv')
            net = subpixel_pre(net, 64, 256, scope='conv')
            net = pixelShuffler(net, scale=2)
            net = prelu_tf(net)

        with tf.variable_scope('output_stage'):
            net = conv2(net, 9, gen_output_channels, 1, scope='conv')

    return net


# Definition of the discriminator
def discriminator(dis_inputs, is_training=True):
    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = conv2(inputs, kernel_size, output_channel, stride, use_bias=False, scope='conv1')
            net = batchnorm(net, is_training)
            net = lrelu(net, 0.2)

        return net

    with tf.device('/gpu:0'):
        with tf.variable_scope('discriminator_unit'):
            # The input layer
            with tf.variable_scope('input_stage'):
                net = conv2(dis_inputs, 3, 64, 1, scope='conv')
                net = lrelu(net, 0.2)

            # The discriminator block part
            # block 1
            net = discriminator_block(net, 64, 3, 2, 'disblock_1')

            # block 2
            net = discriminator_block(net, 128, 3, 1, 'disblock_2')

            # block 3
            net = discriminator_block(net, 128, 3, 2, 'disblock_3')

            # block 4
            net = discriminator_block(net, 256, 3, 1, 'disblock_4')

            # block 5
            net = discriminator_block(net, 256, 3, 2, 'disblock_5')

            # block 6
            net = discriminator_block(net, 512, 3, 1, 'disblock_6')

            # block_7
            net = discriminator_block(net, 512, 3, 2, 'disblock_7')

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
