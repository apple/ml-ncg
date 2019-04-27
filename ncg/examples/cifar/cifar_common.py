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

"""Common utils to builds the CIFAR-10/CIFAR-100 network.

Summary of available functions:

 # Compute input images and labels for training. If you would like to run
 # evaluations, use inputs() instead.
 inputs, labels = distorted_inputs()

"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', '/tmp/cifar_data',
                           """Path to the CIFAR data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

import cifar_input

import cifar_resnet_tf as cifar

# Global constants describing the CIFAR-10/100 data set.
IMAGE_SIZE = cifar_input.IMAGE_SIZE
NUM_CLASSES = cifar_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

DATA_URL_CIFAR10 = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
DATA_URL_CIFAR100 = 'http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'


def distorted_inputs(rank=0):
    """Construct distorted input for CIFAR training using the Reader ops.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    base_dir = FLAGS.data_dir + str(rank)
    if FLAGS.dataset == 'cifar10':
        data_dir = os.path.join(base_dir, 'cifar-10-batches-bin')
    else:
        data_dir = os.path.join(base_dir, 'cifar-100-binary')

    batch_size = FLAGS.batch_size
    images, labels = cifar_input.distorted_inputs(data_dir=data_dir,
                                                  batch_size=batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inputs(eval_data, rank=0):
    """Construct input for CIFAR evaluation using the Reader ops.

    Args:
      eval_data: bool, indicating if one should use the train or eval data set.

    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.

    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')

    base_dir = FLAGS.data_dir + str(rank)

    if FLAGS.dataset == 'cifar10':
        data_dir = os.path.join(base_dir, 'cifar-10-batches-bin')
    else:
        data_dir = os.path.join(base_dir, 'cifar-100-binary')

    batch_size = FLAGS.batch_size
    images, labels = cifar_input.inputs(eval_data=eval_data,
                                        data_dir=data_dir,
                                        batch_size=batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def maybe_download_and_extract(rank=0):
    """Download and extract the tarball from Alex's website."""
    dest_directory = FLAGS.data_dir + str(rank)
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)

    data_url = DATA_URL_CIFAR10
    if FLAGS.dataset == 'cifar100':
        data_url = DATA_URL_CIFAR100

    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

    if FLAGS.dataset == 'cifar10':
        extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
    else:
        extracted_dir_path = os.path.join(dest_directory, 'cifar-100-binary')
    if not os.path.exists(extracted_dir_path):
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def tower_loss(scope, rank=0):
    """Calculate the total loss on a single tower running the CIFAR model.

    Args:
      scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'

    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    # Get images and labels for CIFAR-10.
    with tf.device('/cpu:0'):
        images, labels = distorted_inputs(rank)

    num_classes = 10
    if FLAGS.dataset == 'cifar100':
        num_classes = 100

    images_pl = None
    labels_pl = None

    # Build inference Graph.
    logits = cifar.inference(images, num_classes=num_classes)
    total_loss = cifar.loss(logits, labels)

    # Calculate precision.
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    precision = tf.reduce_mean(tf.to_float(top_k_op))
    tf.summary.scalar('precision', precision)

    return total_loss, images, labels, images_pl, labels_pl, precision


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
