from __future__ import absolute_import, division, print_function

import collections
import math
import os

import numpy as np
import scipy.misc as sic
import tensorflow as tf
from lib.model_gan import generator, discriminator
from lib.model_vgg19 import VGG19_slim
from lib.model_facenet import FaceNet_slim


def SRResnet(inputs, targets, FLAGS):
    # Define the container of the parameter
    crop_size = [int(x) for x in FLAGS.crop_size.split(',')]
    Network = collections.namedtuple('Network', 'content_loss, gen_loss, gen_grads_and_vars, gen_output, train, global_step, \
            learning_rate')
    # Build the generator part
    with tf.variable_scope('generator'):
        output_channel = targets.get_shape().as_list()[-1]
        gen_output = generator(inputs, output_channel, reuse=False, FLAGS=FLAGS)
        gen_output.set_shape([FLAGS.batch_size, crop_size[0]*4, crop_size[1]*4, 3])
    
    # Use the FaceNet feature
    if FLAGS.perceptual_mode == 'FaceNet':
        with tf.name_scope('FaceNet_1') as scope:
            extracted_feature_gen = FaceNet_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
        with tf.name_scope('FaceNet_2') as scope:
            extracted_feature_target = FaceNet_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

    # Use the VGG54 feature
    elif FLAGS.perceptual_mode == 'VGG54':
        with tf.name_scope('vgg19_1') as scope:
            extracted_feature_gen = VGG19_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
        with tf.name_scope('vgg19_2') as scope:
            extracted_feature_target = VGG19_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

    elif FLAGS.perceptual_mode == 'VGG22':
        with tf.name_scope('vgg19_1') as scope:
            extracted_feature_gen = VGG19_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
        with tf.name_scope('vgg19_2') as scope:
            extracted_feature_target = VGG19_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

    elif FLAGS.perceptual_mode == 'MSE':
        extracted_feature_gen = gen_output
        extracted_feature_target = targets

    else:
        raise NotImplementedError('Unknown perceptual type')

    # Calculating the generator loss
    with tf.variable_scope('generator_loss'):
        # Content loss
        with tf.variable_scope('content_loss'):
            # Compute the euclidean distance between the two features
            # check=tf.equal(extracted_feature_gen, extracted_feature_target)
            diff = extracted_feature_gen - extracted_feature_target
            if FLAGS.perceptual_mode == 'MSE':
                content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
            else:
                content_loss = FLAGS.perceptual_scaling * tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))

        gen_loss = content_loss#  + tf.add_n(tf.get_collection('kernel_constrain'))

    # Define the learning rate and global step
    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate,
                                                   staircase=FLAGS.stair)
        incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('generator_train'):
        # Need to wait discriminator to perform train step
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
            gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
            gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)

    # [ToDo] If we do not use moving average on loss??
    exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_averager.apply([content_loss])

    return Network(
        content_loss=exp_averager.average(content_loss),
        gen_loss=gen_loss,
        gen_grads_and_vars=gen_grads_and_vars,
        gen_output=gen_output,
        train=tf.group(update_loss, incr_global_step, gen_train),
        global_step=global_step,
        learning_rate=learning_rate
    )
