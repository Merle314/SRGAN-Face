from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import collections
import os
import math
import scipy.misc as sic
import numpy as np
import glob

# Define the dataloader
def data_loader(FLAGS):
    with tf.device('/cpu:0'):
        # Define the returned data batches
        Data = collections.namedtuple('Data', 'inputs, targets, image_count, steps_per_epoch')
        tfrecords_list = glob.glob(FLAGS.input_dir_tfrecord)
        with tf.variable_scope('load_image'):
            filename_queue = tf.train.string_input_producer(tfrecords_list, shuffle=True)
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
                input_image_HR = tf.subtract(tf.cast(tf.image.decode_png(features["image_raw"], channels=3), tf.float32), 127.5) * 0.0078125
                input_image_HR = tf.reshape(input_image_HR, hr_size)  # The image_shape must be explicitly specified
                input_image_LR = tf.image.resize_images(input_image_HR, lr_size)

            inputs, targets = [input_image_LR, input_image_HR]

        # The data augmentation part
        with tf.name_scope('data_preprocessing'):
            inputs = tf.identity(inputs)
            targets = tf.identity(targets)

            with tf.variable_scope('random_flip'):
                # Check for random flip:
                if (FLAGS.flip is True) and (FLAGS.mode == 'train'):
                    print('[Config] Use random flip')
                    input_images = tf.image.random_flip_left_right(inputs)
                    target_images = tf.image.random_flip_left_right(targets)
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
