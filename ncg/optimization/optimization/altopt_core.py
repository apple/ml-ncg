__all__ = ['lars_grads_and_vars',\
           'reduced_bn_grads_and_vars', 'layer_depth_scaled_grads_and_vars']


import tensorflow as tf

def lars_grads_and_vars(grads_and_vars, lars_lr=0.001):
    """Compute Layer Adaptive Rate Schedule for grads(dir) and multiply the
    learning rate"""
    print('USING LARS WITH SCALE ', lars_lr)
    eps = tf.constant(
        1e-8,
        dtype=tf.float32,
        name='eps'
    )

    lars_g_and_v = []

    for g, v in grads_and_vars:
        g_norm = tf.norm(g)
        g_norm = tf.add(g_norm, eps)
        v_norm = tf.norm(v)
        denom = g_norm
        wd = 0.1*lars_lr*v_norm
        denom = tf.add(denom, wd)

        lars_scale = lars_lr*tf.divide(v_norm, denom)

        new_g = tf.scalar_mul(lars_scale, g)
        l_g_and_v = (new_g, v)

        lars_g_and_v.append(l_g_and_v)

    return lars_g_and_v

def reduced_bn_grads_and_vars(grads_and_vars):
    """Reduce learning rate for grads(dir) of batch norm params"""
    print('SCALING BATCH NORM GRADIENT DOWN BY ', 0.01)

    reduce_bn_g_and_v = []

    for grad, var in grads_and_vars:
        if 'batch_normalization' in var.name or 'BatchNorm' in var.name:
            new_g = tf.scalar_mul(0.01, grad)
        else:
            new_g = grad

        bn_g_and_v = (new_g, var)

        reduce_bn_g_and_v.append(bn_g_and_v)

    return reduce_bn_g_and_v

def layer_depth_scaled_grads_and_vars(grads_and_vars):
    """Compute Layer depth based scaled grads(dir) and multiply the
    learning rate"""

    print('USING LAYER DEPTH SCALING')
    scaled_g_and_v = []

    max_depth = len(grads_and_vars)
    min_lr_scale = 0.1
    max_lr_scale = 1.0
    lr_range = max_lr_scale - min_lr_scale

    for depth, (g, v) in enumerate(grads_and_vars):
        layer_lr_scale = min_lr_scale + lr_range * depth/max_depth

        print ('variable: ', v)
        print ('depth: ', depth, 'max_depth: ', max_depth, 'layer_lr_scale: ', layer_lr_scale)

        new_g = tf.scalar_mul(layer_lr_scale, g)
        l_g_and_v = (new_g, v)

        scaled_g_and_v.append(l_g_and_v)

    return scaled_g_and_v

