"""This module implements the NCG optimization algorithm
__author__ = 'Saurabh Adya'
For licensing see accompanying LICENSE.txt file.
Copyright (C) 2019 Apple Inc. All Rights Reserved.
"""
__all__ = ['AltSyncReplicasOptimizer']

import tensorflow as tf

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import gradients_impl
from tensorflow.python.training import queue_runner

from tensorflow.python.training import sync_replicas_optimizer

from optimization.ncg_core_tfop import NCG_OPTIMIZER_TFOP
from optimization import lars_grads_and_vars
from optimization import layer_depth_scaled_grads_and_vars


class AltSyncReplicasOptimizer(sync_replicas_optimizer.SyncReplicasOptimizer):
    """AltSyncReplicasOptimizer implements Alternate Synchronous Distributed Algorithms.
       The alternate algorithms implemented are Ncg(Ncg w TF ops)
       This interface exists to get the aggregated loss value when using
       virtual batching. Otherwise, it is same as regular SyncReplicasOptimizer
    Args:
       opt: Base TF optimizer (Should be GradientDescentOptimizer with LR=1.0
       replicas_to_aggregate: Number of replicas to aggregate before applying gradient
       total_num_replicas: Total number of replicas in the cluster
       variable_averages: Variable averager
       variables_to_average: List of variables to average
       use_locking: Whether to use Locking before updating the weights
       optimizer: Which alternate optimizer to use (Ncg)
       learning_rate: Base learning rate to use
       ncg_precond: Preconditioner to use in the Ncg algorithm (Bfgs/None)
       ncg_update_rule: Conjugate direction update rule to use in the Ncg algorithm (PR:PolakRibiere / FR:FletcherReaves)
       line_search: Which line search to use: (None/Online/Step)
       step_line_search_period: For Step line_search, the comparison period to use
       line_search_threshold: For line search, what is the threshold (in percent) before we decrease the learning rate.
                              If loss function increase is less than this threshold, then we increase the LR upto the base LR schedule
                              If loss function increase is greater than this threshold, we reduce the LR in proportion to the function increase
       lars_lr: LARS learning rate (<= 0 is Off. Good default value is 0.001)
       verbosity: verbosity level, 0=Default, 1=Info, 2=Debug, 3=Verbose
       name: Name of the alternate optimizer"
    """
    def __init__(self,
                 opt,
                 replicas_to_aggregate,
                 total_num_replicas=None,
                 variable_averages=None,
                 variables_to_average=None,
                 use_locking=False,
                 optimizer='Ncg',
                 learning_rate=0.2,
                 ncg_precond='Bfgs',
                 ncg_update_rule='FR',
                 line_search='Step',
                 step_line_search_period=5,
                 line_search_threshold=1,
                 lars_lr=-1.0,
                 loss=None,
                 verbosity=0,
                 name="Alt_sync_replicas"):
        super(AltSyncReplicasOptimizer, self).__init__(opt=opt,
                                                       replicas_to_aggregate=replicas_to_aggregate,
                                                       total_num_replicas=total_num_replicas,
                                                       variable_averages=variable_averages,
                                                       variables_to_average=variables_to_average,
                                                       use_locking=use_locking,
                                                       name=name)
        self.learning_rate = learning_rate
        self.line_search = line_search
        self.step_line_search_period = step_line_search_period
        self.line_search_threshold = line_search_threshold
        self.ncg_precond = ncg_precond
        self.ncg_update_rule = ncg_update_rule
        self.lars_lr = lars_lr
        self.optimizer = optimizer
        self.verbosity = verbosity
        self.assign_op = None
        self.loss = loss

    def alt_dir_and_vars(self, g_and_v, agg_grads_and_vars, loss, global_step):
        """Given aggregated grads, find a new direction using alternate optimizer
        Args:
          g_and_v: grads_and_vars for the worker
          agg_grads_and_vars: aggregated grads_and_vars from all workers
          loss: scalar loss tensor
          global_step: global_step
        Returns:
          dir_and_vars: Alternate optimizer computed direction and vars
        """
        lr = self.learning_rate

        gs = []
        for g, v in g_and_v:
            g_f = tf.reshape(g, [tf.size(g)])
            gs.append(g_f)

        gs = tf.concat(gs, 0)

        grads = []
        for g, v in agg_grads_and_vars:
            g_f = tf.reshape(g, [tf.size(g)])
            grads.append(g_f)

        grads = tf.concat(grads, 0)
        grads = tf.reshape(grads, gs.get_shape())
        direction = grads

        if self.optimizer == 'Ncg':
            optimizer = NCG_OPTIMIZER_TFOP(grads, precondition=self.ncg_precond,
                                           line_search=self.line_search,
                                           step_line_search_period=self.step_line_search_period,
                                           line_search_threshold=self.line_search_threshold,
                                           update_rule=self.ncg_update_rule,
                                           verbosity=self.verbosity)
            direction, self.assign_op = optimizer.get_direction(grads, loss, lr, global_step)
        else:
            raise Exception('Unsupported Alternate Optimizer {}'.format(self.optimizer))

        direction = tf.reshape(tf.concat(direction, 1), [tf.size(grads)])

        dir_and_vars = []
        start = 0
        for g, v in agg_grads_and_vars:
            g_shape = tf.shape(g)
            shape_length = tf.size(g)
            r_s = shape_length

            d_t = tf.slice(direction, [start], [r_s])

            start = start + r_s

            d_t = tf.reshape(d_t, g_shape)

            dir_and_vars.append((d_t, v))

        return dir_and_vars

    GATE_NONE = 0
    GATE_OP = 1
    GATE_GRAPH = 2

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None):
        """Add operations to minimize `loss` by updating `var_list`.

        This method simply combines calls `compute_gradients()` and
        `apply_gradients()`. If you want to process the gradient before applying
        them call `compute_gradients()` and `apply_gradients()` explicitly instead
        of using this function.

        Args:
        loss: A `Tensor` containing the value to minimize.
        global_step: Optional `Variable` to increment by one after the
        variables have been updated.
        var_list: Optional list of `Variable` objects to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKeys.TRAINABLE_VARIABLES`.
        gate_gradients: How to gate the computation of gradients.  Can be
        `GATE_NONE`, `GATE_OP`, or  `GATE_GRAPH`.
        aggregation_method: Specifies the method used to combine gradient terms.
        Valid values are defined in the class `AggregationMethod`.
        colocate_gradients_with_ops: If True, try colocating gradients with
        the corresponding op.
        name: Optional name for the returned operation.
        grad_loss: Optional. A `Tensor` holding the gradient computed for `loss`.

        Returns:
        An Operation that updates the variables in `var_list`.  If `global_step`
        was not `None`, that operation also increments `global_step`.

        Raises:
        ValueError: If some of the variables are not `Variable` objects.
        """
        grads_and_vars = self.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars], loss))


        return self.apply_gradients(grads_and_vars, global_step=global_step,
                                    name=name)

    def compute_gradients(self, *args, **kwargs):
        """Compute gradients of "loss" for the variables in "var_list".

        This simply wraps the compute_gradients() from the real optimizer. The
        gradients will be aggregated in the apply_gradients() so that user can
        modify the gradients like clipping with per replica global norm if needed.
        The global norm with aggregated gradients can be bad as one replica's huge
        gradients can hurt the gradients from other replicas.

        Args:
        *args: Arguments for compute_gradients().
        **kwargs: Keyword arguments for compute_gradients().

        Returns:
        A list of (gradient, variable) pairs.
        """
        self.loss = args[0]
        return self._opt.compute_gradients(*args, **kwargs)

    def update_loss(self, loss):
        self.loss = loss

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """Apply gradients to variables.

        This contains most of the synchronization implementation and also wraps the
        apply_gradients() from the real optimizer.

        Args:
        grads_and_vars: List of (gradient, variable) pairs as returned by
        compute_gradients().
        global_step: Optional Variable to increment by one after the
        variables have been updated.
        name: Optional name for the returned operation.  Default to the
        name passed to the Optimizer constructor.

        Returns:
        train_op: The op to dequeue a token so the replicas can exit this batch
        and start the next one. This is executed by each replica.

        Raises:
        ValueError: If the grads_and_vars is empty.
        ValueError: If global step is not provided, the staleness cannot be
        checked.
        """
        if not grads_and_vars:
            raise ValueError("Must supply at least one variable")

        if global_step is None:
            raise ValueError("Global step is required to check staleness")

        self._global_step = global_step
        train_ops = []
        aggregated_grad = []
        var_list = []

        loss = self.loss

        self._local_step = variables.Variable(
            initial_value=0,
            trainable=False,
            collections=[ops.GraphKeys.LOCAL_VARIABLES],
            dtype=global_step.dtype.base_dtype,
            name="sync_rep_local_step")
        self.local_step_init_op = state_ops.assign(self._local_step, global_step)
        chief_init_ops = [self.local_step_init_op]
        self.ready_for_local_init_op = variables.report_uninitialized_variables(
            variables.global_variables())

        with ops.name_scope(None, self._name):
            for grad, var in grads_and_vars:
                var_list.append(var)
                with ops.device(var.device):
                    # Dense gradients.
                    if grad is None:
                        aggregated_grad.append(None)  # pass-through.
                        continue
                    elif isinstance(grad, ops.Tensor):
                        grad_accum = data_flow_ops.ConditionalAccumulator(
                            grad.dtype,
                            shape=var.get_shape(),
                            shared_name=var.name + "/grad_accum")
                        train_ops.append(grad_accum.apply_grad(
                            grad, local_step=self._local_step))
                        aggregated_grad.append(grad_accum.take_grad(
                            self._replicas_to_aggregate))
                    else:
                        if not isinstance(grad, ops.IndexedSlices):
                            raise ValueError("Unknown grad type!")
                        grad_accum = data_flow_ops.SparseConditionalAccumulator(
                            grad.dtype, shape=(), shared_name=var.name + "/grad_accum")
                        train_ops.append(grad_accum.apply_indexed_slices_grad(
                            grad, local_step=self._local_step))
                        aggregated_grad.append(grad_accum.take_indexed_slices_grad(
                            self._replicas_to_aggregate))

                    self._accumulator_list.append((grad_accum, var.device))

            aggregated_grads_and_vars = zip(aggregated_grad, var_list)

            with ops.device(global_step.device):
                loss_accum = data_flow_ops.ConditionalAccumulator(
                    loss.dtype,
                    shape=loss.get_shape(),
                    shared_name="loss_accum")
                train_ops.append(loss_accum.apply_grad(
                    loss, local_step=self._local_step))
                aggregated_loss = loss_accum.take_grad(self._replicas_to_aggregate)
                self._accumulator_list.append((loss_accum, global_step.device))

            if self.lars_lr > 0.0:
                with ops.device(global_step.device):
                    aggregated_grads_and_vars = lars_grads_and_vars(aggregated_grads_and_vars, self.lars_lr)

            # Inject NCG apply_gradient step here
            if self.optimizer == 'Ncg':
                # in native tensorflow implementation, the op should run on global_step_device
                with ops.device(global_step.device):
                    aggregated_grads_and_vars = self.alt_dir_and_vars(grads_and_vars, aggregated_grads_and_vars, aggregated_loss, global_step)
            else:
                aggregated_grads_and_vars = self.alt_dir_and_vars(grads_and_vars, aggregated_grads_and_vars, aggregated_loss, global_step)

            # sync_op will be assigned to the same device as the global step.
            with ops.device(global_step.device), ops.name_scope(""):
                update_op = self._opt.apply_gradients(aggregated_grads_and_vars,
                                                      global_step)

            # Create token queue.
            with ops.device(global_step.device), ops.name_scope(""):
                sync_token_queue = (
                    data_flow_ops.FIFOQueue(-1,
                                            global_step.dtype.base_dtype,
                                            shapes=(),
                                            name="sync_token_q",
                                            shared_name="sync_token_q"))
                self._sync_token_queue = sync_token_queue

                # dummy_queue is passed to the queue runner. Don't use the real queues
                # because the queue runner doesn't automatically reopen it once it
                # closed queues in PS devices.
                dummy_queue = (
                    data_flow_ops.FIFOQueue(1,
                                            types_pb2.DT_INT32,
                                            shapes=(),
                                            name="dummy_queue",
                                            shared_name="dummy_queue"))

            with ops.device(global_step.device), ops.name_scope(""):
                # Replicas have to wait until they can get a token from the token queue.
                # train_ops.append(self.assign_op)
                with ops.control_dependencies(train_ops):
                    token = sync_token_queue.dequeue()
                    train_op = state_ops.assign(self._local_step, token)

                update_op_dep = [update_op]
                with ops.control_dependencies(update_op_dep):
                    # Sync_op needs to insert tokens to the token queue at the end of the
                    # step so the replicas can fetch them to start the next step.
                    tokens = array_ops.fill([self._tokens_per_step], global_step)
                    sync_op = sync_token_queue.enqueue_many((tokens,))

                if self._variable_averages is not None:
                    with ops.control_dependencies([sync_op]), ops.name_scope(""):
                        sync_op = self._variable_averages.apply(
                            self._variables_to_average)

                if self.assign_op is not None:
                    with ops.control_dependencies([self.assign_op]), ops.name_scope(""):
                        sync_op = tf.group(sync_op)

                self._chief_queue_runner = queue_runner.QueueRunner(dummy_queue,
                                                                    [sync_op])
            for accum, dev in self._accumulator_list:
                with ops.device(dev):
                    chief_init_ops.append(
                        accum.set_global_step(
                            global_step, name="SetGlobalStep"))
            self.chief_init_op = control_flow_ops.group(*(chief_init_ops))
            self._gradients_applied = True

            return train_op
