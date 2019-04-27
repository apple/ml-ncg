# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10/100 using multiple GPU's with synchronous updates.

Classification Accuracy vs Batch Size vs Optimizers:
Cifar-100 Dataset, Resnet-32 model, 200 epochs
Batch Size  | Accuracy NCG | Accuracy RMS | Accuracy MOM  | Accuracy SGD
--------------------------------------------------------------------
128         | 0.662        | 0.649        | 0.646         | 0.637
256         | 0.684        | 0.651        | 0.649         | 0.638
512         | 0.674        | 0.648        | 0.650         | 0.635
1024        | 0.654        | 0.647        | 0.650         | 0.635
2048        | 0.639        | 0.645        | 0.649         | 0.631
4096        | 0.645        | 0.653        | 0.652         | 0.618
8192        | 0.670        | 0.660        | 0.660         | 0.550
16384       | 0.560        | 0.370        | 0.370         | 0.350

as judged by cifar_eval.py.

Usage:
Please see the README for how to compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from optimization.altopt_tf import AltOptimizer
from six.moves import xrange  # pylint: disable=redefined-builtin

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('num_epochs', 200,
                            """Number of epochs to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('alt_optimizer', 'Ncg',
                           """Whether to run Alternative Optimizer (Ncg/Sgd/Momentum/Rmsprop).""")
tf.app.flags.DEFINE_string('alt_precond', 'Bfgs',
                           """Preconditioning to use in Ncg (None,Bfgs,Diag).""")
tf.app.flags.DEFINE_float('lars_lr', -1.0,
                          'LARS scale in AltOptimizer (Layer Adaptive Rate Schedule.')
tf.app.flags.DEFINE_integer('alt_step_line_search_period', 5,
                            """For Step line_search, comparison period to use in NLCG.""")
tf.app.flags.DEFINE_bool('step_lr', True,
                         'Use official step based LR schedule.')
tf.app.flags.DEFINE_float('alt_line_search_threshold', 1.0,
                          """Line search threshold in NLCG.""")
tf.app.flags.DEFINE_boolean('no_warm_start', False,
                            'Use Warm start to increase learning rate from initial to max in 5 epochs')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial Learning Rate.""")
tf.app.flags.DEFINE_float('max_learning_rate', 0.1,
                          """Max Learning Rate.""")
tf.app.flags.DEFINE_integer('num_warmup_epochs', 25,
                            'Number of Warm Up Epochs.')

import cifar_common

import cifar_resnet_tf as cifar

# Constants dictating the learning rate schedule.
MOMENTUM = 0.9  # Momentum in Momentum optimizer
RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9 # Momentum in RMSProp optimizer
RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.

def alt_optimizer(lr,
                  optimizer='Ncg'):
    """Construct alternate optimizer
    Args:
      lr: learning_rate
      optimizer: Alternate optimizer to construct (Ncg)
    Returns:
      opt: optimizer
    """

    print('Alt Optimizer: ', optimizer)

    if optimizer == 'Sgd':
        opt = tf.train.GradientDescentOptimizer(lr)
    elif optimizer == 'Momentum':
        opt = tf.train.MomentumOptimizer(lr, momentum=MOMENTUM)
    elif optimizer == 'Rmsprop':
        opt = tf.train.RMSPropOptimizer(lr,
                                        RMSPROP_DECAY,
                                        momentum=RMSPROP_MOMENTUM,
                                        epsilon=RMSPROP_EPSILON)
    else:
        opt = AltOptimizer(learning_rate=lr,
                           optimizer=optimizer,
                           ncg_precond=FLAGS.alt_precond,
                           step_line_search_period=FLAGS.alt_step_line_search_period,
                           line_search_threshold=FLAGS.alt_line_search_threshold,
                           lars_lr=FLAGS.lars_lr,
                           name=optimizer)

    return opt

def learning_rate_fn(num_batches_per_epoch, global_step):
    decay_steps = int(num_batches_per_epoch * cifar.NUM_EPOCHS_PER_DECAY)

    if FLAGS.step_lr:
        if FLAGS.num_epochs <= 200:
            boundary_epochs = [102, 153, 204]
        elif FLAGS.num_epochs <= 300:
            boundary_epochs = [153, 229, 306]
        else:
            boundary_epochs = [204, 306, 408]

        decay_rates = [1.0, 0.1, 0.01, 0.001]

        boundaries = [int(num_batches_per_epoch * epoch) for epoch in boundary_epochs]
        print('Learning rate change step boundaries: ', boundaries)
        if not FLAGS.no_warm_start:
            vals = [FLAGS.max_learning_rate * decay for decay in decay_rates]
            max_lr_step = int(FLAGS.num_warmup_epochs * num_batches_per_epoch)

            lr = tf.cond(global_step < max_lr_step,
                         lambda: tf.train.polynomial_decay(FLAGS.initial_learning_rate,
                                                           global_step,
                                                           max_lr_step,
                                                           FLAGS.max_learning_rate,
                                                           power=0.5),
                         lambda: tf.train.piecewise_constant(global_step, boundaries, vals))
        else:
            vals = [FLAGS.initial_learning_rate * decay for decay in decay_rates]
            lr = tf.train.piecewise_constant(global_step, boundaries, vals)
    else:
        # Decay the learning rate exponentially based on the number of steps.
        if not FLAGS.no_warm_start:
            max_lr_step = int(FLAGS.num_warmup_epochs * num_batches_per_epoch)
            lr = tf.cond(global_step < max_lr_step,
                         lambda: tf.train.polynomial_decay(FLAGS.initial_learning_rate,
                                                           global_step,
                                                           max_lr_step,
                                                           FLAGS.max_learning_rate,
                                                           power=0.5),
                         lambda: tf.train.exponential_decay(FLAGS.max_learning_rate,
                                                            global_step,
                                                            decay_steps,
                                                            cifar.LEARNING_RATE_DECAY_FACTOR,
                                                            staircase=True))

            print('INITIAL_LEARNING_RATE: ', FLAGS.initial_learning_rate)
            print('MAX_LEARNING_RATE: ', FLAGS.max_learning_rate)
        else:
            lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                            global_step,
                                            decay_steps,
                                            cifar.LEARNING_RATE_DECAY_FACTOR,
                                            staircase=True)
            print('INITIAL_LEARNING_RATE: ', FLAGS.initial_learning_rate)

    return lr

def train():
    """Train CIFAR-10/100 for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        num_batches_per_epoch = (cifar.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                 (FLAGS.batch_size * FLAGS.num_gpus))
        decay_steps = int(num_batches_per_epoch * cifar.NUM_EPOCHS_PER_DECAY)

        lr = learning_rate_fn(num_batches_per_epoch, global_step)

        if FLAGS.alt_optimizer != '':
            # Create an alternate optimizer
            opt = alt_optimizer(lr, FLAGS.alt_optimizer)
        else:
            # Create an optimizer that performs gradient descent.
            opt = tf.train.GradientDescentOptimizer(lr)

        # Calculate the gradients for each model tower.
        tower_grads = []
        tower_losses = []

        tower_images = []
        tower_labels = []
        tower_images_pl = []
        tower_labels_pl = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (cifar.TOWER_NAME, i)) as scope:
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                            loss, images, labels, images_pl, labels_pl, precision = cifar_common.tower_loss(scope)
                        tower_losses.append(loss)
                        tower_images.append(images)
                        tower_labels.append(labels)
                        tower_images_pl.append(images_pl)
                        tower_labels_pl.append(labels_pl)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Retain the summaries from the final tower.
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Retain the Batch Normalization updates operations only from the
                        # final tower. Ideally, we should grab the updates from all towers
                        # but these stats accumulate extremely fast so we can ignore the
                        # other stats from the other towers without significant detriment.
                        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope)
                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = cifar_common.average_gradients(tower_grads)

        loss = tf.add_n(tower_losses)
        loss = tf.divide(loss, FLAGS.num_gpus)

        # Add a summary to track the learning rate.
        summaries.append(tf.summary.scalar('learning_rate', lr))

        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        batchnorm_updates_op = tf.group(*batchnorm_updates)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables() +
                                                        tf.moving_average_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        max_steps = int(FLAGS.num_epochs * num_batches_per_epoch)

        print('Max Training Steps: ', max_steps)

        for step in xrange(max_steps):
            start_time = time.time()

            _, loss_value, lrate = sess.run([train_op, loss, lr])

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                format_str = ('%s: step %d, loss = %.2f, lrate = %.4f, (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value, lrate,
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        return loss_value


def main(argv=None):  # pylint: disable=unused-argument
    cifar_common.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
