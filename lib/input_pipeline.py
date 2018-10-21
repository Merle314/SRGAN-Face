from __future__ import absolute_import, division, print_function

import collections
import glob
import math
import os

import numpy as np
import scipy.misc as sic
import tensorflow as tf
from lib.ops import random_flip


# Define the dataloader
def data_loader(FLAGS):
    with tf.device('/cpu:0'):
        # Define the returned data batches
        Data = collections.namedtuple(
            'Data', 'inputs, targets, image_count, steps_per_epoch')
        tfrecords_list = glob.glob(FLAGS.input_dir)
        with tf.variable_scope('load_image'):
            filename_queue = tf.train.string_input_producer(
                tfrecords_list, shuffle=True)
            with tf.variable_scope('read_parse_preproc'):
                reader = tf.TFRecordReader()
                key, records = reader.read(filename_queue)
                lr_size = [int(x) for x in FLAGS.crop_size.split(',')]
                hr_size = [lr_size[0]*4, lr_size[1]*4, 3]
                # parse records
                features = tf.parse_single_example(
                    records,
                    features={
                        # "image_LR": tf.FixedLenFeature([], tf.string),
                        # 'label': tf.FixedLenFeature([], tf.int64),
                        "image_raw": tf.FixedLenFeature([], tf.string)
                    }
                )
                # input_image_HR = tf.image.decode_png(features["image_raw"], channels=3)
                input_image_HR = tf.decode_raw(features["image_raw"], tf.uint8)
                # print(input_image_HR)
                # The image_shape must be explicitly specified
                input_image_HR = tf.reshape(input_image_HR, hr_size)
                input_image_LR = tf.cast(tf.image.resize_images(
                    input_image_HR, lr_size), tf.uint8)
                input_image_LR = tf.image.convert_image_dtype(
                    input_image_LR, dtype=tf.float32)
                input_image_HR = tf.image.convert_image_dtype(
                    input_image_HR, dtype=tf.float32)
            inputs, targets = [input_image_LR, input_image_HR]
        # The data augmentation part
        with tf.name_scope('data_preprocessing'):
            inputs = inputs*2.0-1.0
            targets = targets*2.0-1.0

            with tf.variable_scope('random_flip'):
                # Check for random flip:
                if (FLAGS.flip is True):
                    print('[Config] Use random flip')
                    decision = tf.random_uniform([], 0, 1, dtype=tf.float32)
                    input_images = random_flip(inputs, decision)
                    target_images = random_flip(targets, decision)
                else:
                    input_images = tf.identity(inputs)
                    target_images = tf.identity(targets)

        inputs_batch, targets_batch = tf.train.shuffle_batch([input_images, target_images],
                                                             batch_size=FLAGS.batch_size, capacity=FLAGS.image_queue_capacity+4*FLAGS.batch_size,
                                                             min_after_dequeue=FLAGS.image_queue_capacity, num_threads=FLAGS.queue_thread)

        steps_per_epoch = int(math.ceil(FLAGS.image_count / FLAGS.batch_size))
    return Data(
        inputs=inputs_batch,
        targets=targets_batch,
        image_count=FLAGS.image_count,
        steps_per_epoch=steps_per_epoch
    )
