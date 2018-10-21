from __future__ import absolute_import, division, print_function

import math
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.input_pipeline import data_loader
from lib.ops import print_configuration_op, compute_psnr
from lib.SRGAN import SRGAN

Flags = tf.app.flags

# The system parameter
Flags.DEFINE_string('output_dir', '/media/lab225/Document2/merle/train_result/gan_face', 'The output directory of the checkpoint')
Flags.DEFINE_string('summary_dir', '/media/lab225/Document2/merle/train_result/gan_face/log', 'The dirctory to output the summary')
Flags.DEFINE_string('checkpoint', '/media/lab225/Document2/merle/train_result/gan/model-30000', 'If provided, the weight will be restored from the provided checkpoint')
Flags.DEFINE_boolean('pre_trained_model', True, 'If set True, the weight will be loaded but the global_step will still '
                                                 'be 0. If set False, you are going to continue the training. That is, '
                                                 'the global_step will be initiallized from the checkpoint, too')
Flags.DEFINE_string('pre_trained_model_type', 'SRResnet', 'The type of pretrained model (SRGAN or SRResnet)')
Flags.DEFINE_string('perceptual_ckpt', '/media/lab225/Documents/merle/faceDataSet/Models/20181007-144210/model-20181007-144210.ckpt-79000', 'path to checkpoint file for the perceptual model')
# The data preparing operation
Flags.DEFINE_integer('batch_size', 48, 'Batch size of the input batch')
Flags.DEFINE_string('input_dir', '/media/lab225/Document2/merle/faceDataset/celeba_align_112x96_tfrecord/*.tfrecord', 'The directory of the input tfrecord data dir')
Flags.DEFINE_boolean('flip', True, 'Whether random flip data augmentation is applied')
Flags.DEFINE_boolean('random_crop', True, 'Whether perform the random crop')
Flags.DEFINE_string('crop_size', '28,24', 'The crop size of the training image')
Flags.DEFINE_integer('name_queue_capacity', 2048, 'The capacity of the filename queue (suggest large to ensure'
                                                  'enough random shuffle.')
Flags.DEFINE_integer('image_queue_capacity', 2048, 'The capacity of the image queue (suggest large to ensure'
                                                   'enough random shuffle')
Flags.DEFINE_integer('queue_thread', 10, 'The threads of the queue (More threads can speedup the training process.')
Flags.DEFINE_integer('image_count', 200000, 'The total image numbers in tfrecord files.')
# Generator configuration
Flags.DEFINE_integer('num_resblock', 16, 'How many residual blocks are there in the generator')
# The content loss parameter
Flags.DEFINE_string('perceptual_mode', 'FaceNet', 'VGG54, VGG18, FaceNet, The type of feature used in perceptual loss')
Flags.DEFINE_string('perceptual_scope', 'InceptionResnetV1', 'vgg_19, InceptionResnetV1, Resface')
Flags.DEFINE_float('EPS', 1e-10, 'The eps added to prevent nan')
Flags.DEFINE_float('ratio', 0.1, 'The ratio between content loss and adversarial loss')
Flags.DEFINE_float('perceptual_scaling', 0.4, 'The scaling factor for the perceptual loss if using vgg perceptual loss')
# The training parameters
Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 40000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 200, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')


FLAGS = Flags.FLAGS

# Print the configuration of the model
print_configuration_op(FLAGS)

# Check the output_dir is given
if FLAGS.output_dir is None:
    raise ValueError('The output directory is needed')

# Check the output directory to save the checkpoint
if not os.path.exists(FLAGS.output_dir):
    os.mkdir(FLAGS.output_dir)

# Check the summary directory to save the event
if not os.path.exists(FLAGS.summary_dir):
    os.mkdir(FLAGS.summary_dir)

# Load data for training and testing
# ToDo Add online downscaling
data = data_loader(FLAGS)
print('Data count = %d' % (data.image_count))

# Connect to the network
Net = SRGAN(data.inputs, data.targets, FLAGS)
print('Finish building the network!!!')

# Convert the images output from the network
with tf.name_scope('convert_image'):
    # Deprocess the images outputed from the model
    inputs = (data.inputs+1.0)/2.0
    targets = (data.targets+1.0)/2.0
    outputs = (Net.gen_output+1.0)/2.0

    # Convert back to uint8
    converted_inputs = tf.image.convert_image_dtype(inputs, dtype=tf.uint8, saturate=True)
    converted_targets = tf.image.convert_image_dtype(targets, dtype=tf.uint8, saturate=True)
    converted_outputs = tf.image.convert_image_dtype(outputs, dtype=tf.uint8, saturate=True)

# Compute PSNR
with tf.name_scope("compute_psnr"):
    psnr = compute_psnr(converted_targets, converted_outputs)

# Add image summaries
with tf.name_scope('inputs_summary'):
    tf.summary.image('input_summary', converted_inputs)

with tf.name_scope('targets_summary'):
    tf.summary.image('target_summary', converted_targets)

with tf.name_scope('outputs_summary'):
    tf.summary.image('outputs_summary', converted_outputs)

# Add scalar summary
tf.summary.scalar('discriminator_loss', Net.discrim_loss)
tf.summary.scalar('adversarial_loss', Net.adversarial_loss)
tf.summary.scalar('content_loss', Net.content_loss)
tf.summary.scalar('generator_loss', Net.content_loss + FLAGS.ratio*Net.adversarial_loss)
tf.summary.scalar('PSNR', psnr)
tf.summary.scalar('learning_rate', Net.learning_rate)

# Define the saver and weight initiallizer
saver = tf.train.Saver(max_to_keep=1)

# The variable list
var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
# Here if we restore the weight from the SRResnet the var_list2 do not need to contain the discriminator weights
# On contrary, if you initial your weight from other SRGAN checkpoint, var_list2 need to contain discriminator
# weights.
if FLAGS.pre_trained_model_type == 'SRGAN':
    var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
elif FLAGS.pre_trained_model_type == 'SRResnet':
    var_list_temp = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    # for var in var_list_temp:
    #     print(var.name)
    # exclusions = ['generator/generator_unit/subpixelconv_stage1/conv/kernel:0', 'generator/generator_unit/subpixelconv_stage2/conv/kernel:0']
    exclusions = []
    var_list2 = [v for v in var_list_temp if v.name not in exclusions]
else:
    raise ValueError('Unknown pre_trained model type!!')

weight_initiallizer = tf.train.Saver(var_list2)

# When using MSE loss, no need to restore the vgg net
if not FLAGS.perceptual_mode == 'MSE':
    perceptual_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=FLAGS.perceptual_scope)
    perceptual_restore = tf.train.Saver(perceptual_var_list)

# Start the session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# Use superviser to coordinate all queue and summary writer
sv = tf.train.Supervisor(logdir=FLAGS.summary_dir, save_summaries_secs=0, saver=None)
with sv.managed_session(config=config) as sess:
    if (FLAGS.checkpoint is not None) and (FLAGS.pre_trained_model is False):
        print('Loading model from the checkpoint...')
        checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint)
        saver.restore(sess, checkpoint)

    elif (FLAGS.checkpoint is not None) and (FLAGS.pre_trained_model is True):
        print('Loading weights from the pre-trained model')
        weight_initiallizer.restore(sess, FLAGS.checkpoint)

    if not FLAGS.perceptual_mode == 'MSE':
        perceptual_restore.restore(sess, FLAGS.perceptual_ckpt)
        print('Perceptual model:', FLAGS.perceptual_mode, 'restored successfully!!')

    # Performing the training
    if FLAGS.max_epoch is None:
        if FLAGS.max_iter is None:
            raise ValueError('one of max_epoch or max_iter should be provided')
        else:
            max_iter = FLAGS.max_iter
    else:
        max_iter = FLAGS.max_epoch * data.steps_per_epoch

    print('Optimization starts!!!')
    start = time.time()
    for step in range(max_iter):
        fetches = {
            "train": Net.train,
            "global_step": sv.global_step,
        }

        if ((step+1) % FLAGS.display_freq) == 0:
            fetches["discrim_loss"] = Net.discrim_loss
            fetches["adversarial_loss"] = Net.adversarial_loss
            fetches["content_loss"] = Net.content_loss
            fetches["PSNR"] = psnr
            fetches["learning_rate"] = Net.learning_rate
            fetches["global_step"] = Net.global_step
        if ((step+1) % FLAGS.summary_freq) == 0:
            fetches["summary"] = sv.summary_op

        results = sess.run(fetches)

        if ((step + 1) % FLAGS.summary_freq) == 0:
            print('Recording summary!!')
            sv.summary_writer.add_summary(results['summary'], results['global_step'])

        if ((step + 1) % FLAGS.display_freq) == 0:
            train_epoch = math.ceil(results["global_step"] / data.steps_per_epoch)
            train_step = (results["global_step"] - 1) % data.steps_per_epoch + 1
            rate = (step + 1) * FLAGS.batch_size / (time.time() - start)
            remaining = (max_iter - step) * FLAGS.batch_size / rate
            print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
            print("global_step", results["global_step"])
            print("PSNR", results["PSNR"])
            print("discrim_loss", results["discrim_loss"])
            print("adversarial_loss", results["adversarial_loss"])
            print("content_loss", results["content_loss"])
            print("learning_rate", results['learning_rate'])

        if ((step +1) % FLAGS.save_freq) == 0:
            print('Save the checkpoint')
            saver.save(sess, os.path.join(FLAGS.output_dir, 'model'), global_step=sv.global_step)

    print('Optimization done!!!!!!!!!!!!')
