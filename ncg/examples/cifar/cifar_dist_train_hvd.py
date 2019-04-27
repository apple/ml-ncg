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

"""A binary to train CIFAR-10/CIFAR-100 using distributed multiple GPU's with synchronous updates.

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
Please see the README on how to compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time
import os
import sys
import math
import pickle

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_name = dir_path + "/inception"
sys.path.append(dir_name)

import horovod.tensorflow as hvd

from optimization.altopt_tf import *
from optimization.altopt_syncdist import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('num_tries', 1,
                            """Number of trials to run.""")
tf.app.flags.DEFINE_string('train_dir', 'cifar_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_integer('num_eval_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_integer('num_epochs', 200,
                            """Number of epochs to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

tf.app.flags.DEFINE_string('alt_sync_optimizer', 'Ncg',
                           'Use Alternate optimizer (Ncg or Sgd or Momentum or Rmsprop).')
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_float('lars_lr', -1.0,
                          'LARS scale in AltOptimizer (Layer Adaptive Rate Schedule. Does not work for Hfcg AltOptimizer')
tf.app.flags.DEFINE_string('alt_line_search', 'Step',
                           'Linesearch to use in alt_optimizer (None/Step/Online).')
tf.app.flags.DEFINE_string('alt_ncg_update_rule', 'FR',
                           'Update rule to use in NLCG (FR,PR).')
tf.app.flags.DEFINE_string('alt_precond', 'Bfgs',
                           'Preconditioning to use in Ncg (None/Bfgs).')
tf.app.flags.DEFINE_boolean('no_warm_start', False,
                            'Use Warm start to increase learning rate from initial to max in 5 epochs')
tf.app.flags.DEFINE_bool('step_lr', True,
                         'Use official step based LR schedule.')
tf.app.flags.DEFINE_integer('alt_step_line_search_period', 5,
                          """For Step line_search, comparison period to use in NLCG.""")
tf.app.flags.DEFINE_float('alt_line_search_threshold', 1.0,
                          """Line search threshold in NLCG.""")
tf.app.flags.DEFINE_integer('num_warmup_epochs', 25,
                            'Number of Warm Up Epochs.')
tf.app.flags.DEFINE_integer('save_interval_secs', 5 * 60,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 120,
                            'Save summaries interval seconds.')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          """Initial Learning Rate.""")
tf.app.flags.DEFINE_float('max_learning_rate', 0.1,
                          """Max Learning Rate.""")

import cifar_common

import cifar_resnet_tf as cifar

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
MOMENTUM = 0.9  # Momentum in RMSProp/Momentum.
RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.

def alt_sync_optimizer(lr,
                       num_replicas_to_aggregate,
                       num_workers,
                       exp_moving_averager=None,
                       variables_to_average=None,
                       optimizer='Ncg'):
    print('Synchronous optimizer used is: ', optimizer)

    if optimizer == 'Sgd':
        opt = tf.train.GradientDescentOptimizer(lr)
        # Create synchronous replica optimizer.
        opt = tf.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=num_replicas_to_aggregate,
            total_num_replicas=num_workers,
            variable_averages=exp_moving_averager,
            variables_to_average=variables_to_average)
    elif optimizer == 'Momentum':
        opt = tf.train.MomentumOptimizer(lr, momentum=MOMENTUM)
        # Create synchronous replica optimizer.
        opt = tf.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=num_replicas_to_aggregate,
            total_num_replicas=num_workers,
            variable_averages=exp_moving_averager,
            variables_to_average=variables_to_average)
    elif optimizer == 'Rmsprop':
        opt = tf.train.RMSPropOptimizer(lr,
                                        RMSPROP_DECAY,
                                        momentum=MOMENTUM,
                                        epsilon=RMSPROP_EPSILON)
        # Create synchronous replica optimizer.
        opt = tf.train.SyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=num_replicas_to_aggregate,
            total_num_replicas=num_workers,
            variable_averages=exp_moving_averager,
            variables_to_average=variables_to_average)
    elif optimizer == 'Ncg':
        # Create an optimizer that performs gradient descent with LR=1.0.
        opt = tf.train.GradientDescentOptimizer(1.0)

        # Create synchronous replica optimizer.
        opt = AltSyncReplicasOptimizer(
            opt,
            replicas_to_aggregate=num_replicas_to_aggregate,
            total_num_replicas=num_workers,
            variable_averages=exp_moving_averager,
            variables_to_average=variables_to_average,
            use_locking=False,
            optimizer=optimizer,
            learning_rate=lr,
            ncg_precond=FLAGS.alt_precond,
            ncg_update_rule=FLAGS.alt_ncg_update_rule,
            line_search=FLAGS.alt_line_search,
            step_line_search_period=FLAGS.alt_step_line_search_period,
            line_search_threshold=FLAGS.alt_line_search_threshold,
            lars_lr=FLAGS.lars_lr)
    else:
        raise Exception('Unsupported Alternate Optimizer {}'.format(optimizer))

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


def train(num_try=0):
    """Train CIFAR-10/CIFAR-100 for a number of steps."""

    train_dir = FLAGS.train_dir + '_' + str(num_try)

    num_workers = hvd.size()
    task_id = hvd.rank()
    if task_id == 0:
        if tf.gfile.Exists(train_dir):
            tf.gfile.DeleteRecursively(train_dir)
        tf.gfile.MakeDirs(train_dir)

    # If no value is given, num_replicas_to_aggregate defaults to be the number of
    # workers.

    print('NUM_WORKERS:', num_workers, 'NUM_GPUS:', FLAGS.num_gpus,
          'BATCH_SIZE_PER_WORKER: ', FLAGS.batch_size * FLAGS.num_gpus)

    g = tf.Graph()
    with g.as_default():
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False)

        # Calculate the learning rate schedule.
        num_replicas_to_aggregate = -1
        if FLAGS.num_replicas_to_aggregate == -1:
            num_replicas_to_aggregate = num_workers
        else:
            num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate

        num_batches_per_epoch = (cifar.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                                 (FLAGS.batch_size * FLAGS.num_gpus * num_workers * num_replicas_to_aggregate))

        lr = learning_rate_fn(num_batches_per_epoch, global_step)

        # Create optimizer.
        opt1 = alt_sync_optimizer(lr,
                                  num_replicas_to_aggregate=num_replicas_to_aggregate,
                                  num_workers=1,
                                  optimizer=FLAGS.alt_sync_optimizer)
        opt = hvd.DistributedOptimizer(opt1)

        # Calculate the gradients for each model tower.
        tower_grads = []
        tower_losses = []

        tower_images = []
        tower_labels = []
        tower_images_pl = []
        tower_labels_pl = []
        tower_precision = []

        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (cifar.TOWER_NAME, i)) as scope:
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                            loss, images, labels, images_pl, labels_pl, precision = cifar_common.tower_loss(scope, rank=hvd.rank())
                        tower_losses.append(loss)
                        tower_images.append(images)
                        tower_labels.append(labels)
                        tower_images_pl.append(images_pl)
                        tower_labels_pl.append(labels_pl)
                        tower_precision.append(precision)

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
        loss = hvd.allreduce(loss)

        precision = tf.add_n(tower_precision)
        precision = tf.divide(precision, FLAGS.num_gpus)
        precision = hvd.allreduce(precision)

        tf.summary.scalar('train_precision', precision)

        #Update the loss to averaged loss across all workers
        if FLAGS.alt_sync_optimizer == 'Ncg':
            opt1.update_loss(loss)

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
        saver = tf.train.Saver(save_relative_paths=True, sharded=False,
                               max_to_keep=20)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init_op = tf.global_variables_initializer()
        local_init_op = tf.local_variables_initializer()

        max_steps = int(FLAGS.num_epochs * num_batches_per_epoch)

        print('Max Training Steps: ', max_steps)

        inter_op_parallelism_threads = 4 * FLAGS.num_gpus
        intra_op_parallelism_threads = 4 * FLAGS.num_gpus

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement,
            inter_op_parallelism_threads=inter_op_parallelism_threads,
            intra_op_parallelism_threads=intra_op_parallelism_threads)

        visible_device = FLAGS.num_gpus * hvd.local_rank()
        visible_device_str = str(visible_device)
        for i in range(1, FLAGS.num_gpus):
            visible_device = visible_device + 1
            visible_device_str = visible_device_str + ',' + str(visible_device)

        print('Visible GPUS: ', visible_device_str)
        sess_config.gpu_options.visible_device_list = visible_device_str
        sess_config.gpu_options.allow_growth = True

        scaffold = tf.train.Scaffold(init_op=init_op,
                                     local_init_op=local_init_op,
                                     saver=saver,
                                     summary_op=summary_op)

        hooks = [
            hvd.BroadcastGlobalVariablesHook(0),
            tf.train.StopAtStepHook(last_step=max_steps)
        ]

        sync_replicas_hook = opt1.make_session_run_hook(is_chief=True)
        hooks.append(sync_replicas_hook)

        if task_id == 0:
            checkpoint_dir = train_dir
        else:
            checkpoint_dir = None

        with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                               hooks=hooks,
                                               config=sess_config,
                                               save_summaries_secs=FLAGS.save_summaries_secs,
                                               save_checkpoint_secs=FLAGS.save_interval_secs,
                                               scaffold=scaffold,
                                               stop_grace_period_secs=10) as sess:

            while not sess.should_stop():
                start_time = time.time()

                _, loss_value, lrate, step, precision_value = sess.run([train_op, loss, lr, global_step, precision])

                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if task_id == 0 and step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus * hvd.size()
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration

                    format_str = (
                    '%s: step %d, train_precision = %.2f, loss = %.2f, lrate = %.4f, (%.1f examples/sec; %.3f '
                    'sec/batch)')
                    print(format_str % (datetime.now(), step, precision_value, loss_value,
                                        lrate, examples_per_sec, sec_per_batch))

    return loss_value


def eval_once(saver, summary_writer, top_k_op, summary_op, num_try):
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
      num_try: trial number
    """
    with tf.Session() as sess:
        train_dir = FLAGS.train_dir + '_' + str(num_try)
        ckpt = tf.train.get_checkpoint_state(train_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Loaded checkpoint: ', ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_eval_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

        return precision


def evaluate(eval_data='test', num_try=0):
    """Eval CIFAR-10/CIFAR-100 for a number of steps."""
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-100.
        eval_data = eval_data == 'test'
        images, labels = cifar_common.inputs(eval_data=eval_data)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        num_classes = 10
        if FLAGS.dataset == 'cifar100':
            num_classes = 100

        logits = cifar.inference(images, num_classes=num_classes, for_training=False)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        precision = eval_once(saver, summary_writer, top_k_op, summary_op, num_try)

        summary_writer.close()

        return precision


def main(_):
    hvd.init()
    rank = hvd.rank()
    tf.set_random_seed(rank)

    cifar_common.maybe_download_and_extract(rank)

    train_losses = []
    test_precisions = []
    train_precisions = []
    for i in range(FLAGS.num_tries):
        train_loss = train(i)

        if rank == 0:
            precision_test = evaluate('test', i)
            precision_train = evaluate('train', i)

            print('Experiment ', i, ' Train loss: ', train_loss, ' Test Precision: ', precision_test,
                  ' Train Precision: ', precision_train)

            train_losses.append(train_loss)
            test_precisions.append(precision_test)
            train_precisions.append(precision_train)

    if rank == 0:
        mean_train_loss = np.mean(train_losses)
        mean_test_precision = np.mean(test_precisions)
        mean_train_precision = np.mean(train_precisions)

        stddev_train_loss = np.std(train_losses)
        stddev_test_precision = np.std(test_precisions)
        stddev_train_precision = np.std(train_precisions)

        print('Train Losses: ', train_losses)
        print('Test Precision: ', test_precisions)
        print('Train Precision: ', train_precisions)

        print('Mean Train loss: ', mean_train_loss, 'Mean Test Precision: ', mean_test_precision,
              'Mean Train Precision: ', mean_train_precision)
        print('StdDev Train loss: ', stddev_train_loss, 'StdDev Test Precision: ', stddev_test_precision,
              'StdDev Train Precision: ', stddev_train_precision)

        train_dir = FLAGS.train_dir + '_' + str(0)

        PIK = train_dir + "/pickle.dat"
        data = [train_losses, test_precisions, train_precisions]
        with open(PIK, "wb") as f:
            pickle.dump(data, f)

        with open(PIK, "rb") as f:
            print('Pickled Experiment Data:')
            data = pickle.load(f)
            print(data)


if __name__ == '__main__':
    tf.app.run()
