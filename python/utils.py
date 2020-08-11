import tensorflow as tf
import numpy as np

def leaky_relu(input, alpha=0.02):
    return tf.maximum(input, tf.minimum(alpha * input, 0))


def leaky_relu2(input, alpha=0.02):
    return tf.maximum(tf.minimum(input, 1), tf.maximum(alpha * input, -1))


def batch_norm_wrapper(inputs, name='batch_norm', is_training=False, decay=0.9, epsilon=1e-5):
    with tf.variable_scope(name) as scope:
        if is_training == True:
            scale = tf.get_variable('scale', dtype=tf.float32, trainable=True, initializer=tf.ones([inputs.get_shape()[-1]], dtype=tf.float32))
            beta = tf.get_variable('beta', dtype=tf.float32, trainable=True, initializer=tf.zeros([inputs.get_shape()[-1]], dtype=tf.float32))
            pop_mean = tf.get_variable('overallmean', dtype=tf.float32, trainable=False, initializer=tf.zeros([inputs.get_shape()[-1]], dtype=tf.float32))
            pop_var = tf.get_variable('overallvar', dtype=tf.float32, trainable=False, initializer=tf.ones([inputs.get_shape()[-1]], dtype=tf.float32))
        else:
            scope.reuse_variables()
            scale = tf.get_variable('scale', dtype=tf.float32, trainable=True)
            beta = tf.get_variable('beta', dtype=tf.float32, trainable=True)
            pop_mean = tf.get_variable('overallmean', dtype=tf.float32, trainable=False)
            pop_var = tf.get_variable('overallvar', dtype=tf.float32, trainable=False)

        if is_training:
            axis = list(range(len(inputs.get_shape()) - 1))
            batch_mean, batch_var = tf.nn.moments(inputs, axis)
            train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
            train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)