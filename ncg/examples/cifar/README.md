* [CIFAR-10/100 is a common benchmark in machine learning for image recognition.](http://www.cs.toronto.edu/~kriz/cifar.html)

* Code in this directory demonstrates how to use TensorFlow to train and evaluate a convolutional neural network (CNN) on both CPU and GPU. We also demonstrate how to train a CNN over multiple GPUs.

* Detailed instructions on how to get started available at [tensorflow website](http://tensorflow.org/tutorials/deep_cnn/)



### Usage examples for multi-gpu  training
```
 python cifar_multi_gpu_train.py --batch_size 128 --num_gpus 2 --alt_optimizer Momentum --dataset cifar10
 
 CUDA_VISIBLE_DEVICES=0,1 python cifar_multi_gpu_train.py --batch_size 1024 --max_learning_rate 0.8 --alt_optimizer Ncg --num_gpus 2 --dataset cifar100

 CUDA_VISIBLE_DEVICES=0,1 python cifar_multi_gpu_train.py --batch_size 1024 --max_learning_rate 0.8 --alt_optimizer Ncg --lars_lr 0.001 --num_gpus 2 --dataset cifar100

```
### Usage examples for Distributed Tensorflow with Horovod

```
  CUDA_VISIBLE_DEVICES=0,1 mpirun --np 2 python cifar_dist_train_hvd.py --alt_sync_optimizer Ncg --batch_size 1024 --max_learning_rate 1.6 --initial_learning_rate 0.001 --num_replicas_to_aggregate 2 --dataset cifar100
```
