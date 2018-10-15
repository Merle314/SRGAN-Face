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


# Define the whole network architecture
def SRGAN(inputs, targets, FLAGS):
    # Calculate the crop_size
    crop_size = [int(x) for x in FLAGS.crop_size.split(',')]
    # Define the container of the parameter
    Network = collections.namedtuple('Network', 'discrim_real_output, discrim_fake_output, discrim_loss, \
        discrim_grads_and_vars, adversarial_loss, content_loss, gen_grads_and_vars, gen_output, train, global_step, \
        learning_rate')

    # Build the generator part
    with tf.variable_scope('generator'):
        output_channel = targets.get_shape().as_list()[-1]
        gen_output = generator(inputs, output_channel, reuse=False, FLAGS=FLAGS)
        gen_output.set_shape([FLAGS.batch_size, crop_size[0]*4, crop_size[1]*4, 3])

    # Build the fake discriminator
    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse=False):
            discrim_fake_output = discriminator(gen_output, is_training=True)

    # Build the real discriminator
    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator', reuse=True):
            discrim_real_output = discriminator(targets, is_training=True)

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

    # Use the VGG22 feature
    elif FLAGS.perceptual_mode == 'VGG22':
        with tf.name_scope('vgg19_1') as scope:
            extracted_feature_gen = VGG19_slim(gen_output, FLAGS.perceptual_mode, reuse=False, scope=scope)
        with tf.name_scope('vgg19_2') as scope:
            extracted_feature_target = VGG19_slim(targets, FLAGS.perceptual_mode, reuse=True, scope=scope)

    # Use MSE loss directly
    elif FLAGS.perceptual_mode == 'MSE':
        extracted_feature_gen = gen_output
        extracted_feature_target = targets

    else:
        raise NotImplementedError('Unknown perceptual type!!')

    # Calculating the generator loss
    with tf.variable_scope('generator_loss'):
        # Content loss
        with tf.variable_scope('content_loss'):
            # Compute the euclidean distance between the two features
            diff = extracted_feature_gen - extracted_feature_target
            if FLAGS.perceptual_mode == 'MSE':
                content_loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))
            else:
                content_loss = FLAGS.perceptual_scaling*tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[3]))

        with tf.variable_scope('adversarial_loss'):
            adversarial_loss = tf.reduce_mean(-tf.log(discrim_fake_output + FLAGS.EPS))

        gen_loss = content_loss + (FLAGS.ratio)*adversarial_loss
        print(adversarial_loss.get_shape())
        print(content_loss.get_shape())

    # Calculating the discriminator loss
    with tf.variable_scope('discriminator_loss'):
        discrim_fake_loss = tf.log(1 - discrim_fake_output + FLAGS.EPS)
        discrim_real_loss = tf.log(discrim_real_output + FLAGS.EPS)

        discrim_loss = tf.reduce_mean(-(discrim_fake_loss + discrim_real_loss))

    # Define the learning rate and global step
    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, FLAGS.decay_step, FLAGS.decay_rate, staircase=FLAGS.stair)
        incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('dicriminator_train'):
        discrim_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        discrim_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
        discrim_grads_and_vars = discrim_optimizer.compute_gradients(discrim_loss, discrim_tvars)
        discrim_train = discrim_optimizer.apply_gradients(discrim_grads_and_vars)

    with tf.variable_scope('generator_train'):
        # Need to wait discriminator to perform train step
        with tf.control_dependencies([discrim_train]+ tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gen_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=FLAGS.beta)
            gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
            gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)

    #[ToDo] If we do not use moving average on loss??
    exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_averager.apply([discrim_loss, adversarial_loss, content_loss])

    return Network(
        discrim_real_output=discrim_real_output,
        discrim_fake_output=discrim_fake_output,
        discrim_loss=exp_averager.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        adversarial_loss=exp_averager.average(adversarial_loss),
        content_loss=exp_averager.average(content_loss),
        gen_grads_and_vars=gen_grads_and_vars,
        gen_output=gen_output,
        train=tf.group(update_loss, incr_global_step, gen_train),
        global_step=global_step,
        learning_rate=learning_rate
    )
