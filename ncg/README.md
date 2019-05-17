# NCG Usage Guide

For more details, please refer to [Nonlinear Conjugate Gradients For Scaling Synchronous Distributed DNN Training](https://arxiv.org/abs/1812.02886)

## Optimization package
Optimization module implements the second order Non Linear Conjugate Gradient DNN optimization methods.

### Package contents
- **ncg_core_tfop.py** : Native Tensorflow op based implementation of Non-linear conjugate gradient optimizer
- **altopt_tf.py** : Harness to be used for a single machine/asynchronous distributed optimizer for ncg_core
- **altopt_syncdist.py** : Harness to be used to Synchronous distributed optimizer for ncg_core

## Examples of using NCG
### Synchronous Distributed Single machine / Horovod
  * Sample code at *[examples/cifar/README.md](examples/cifar/README.md)*
  * Use AltOptimizer defined in altopt_tf.py as the optimizer

### Synchronous Distributed with Replica Optimizer (for virtual batching)
  * Sample code at *[examples/cifar/README.md](examples/cifar/README.md)*
  * Use AltSyncReplicasOptimizer defined in altopt_syncdist.py