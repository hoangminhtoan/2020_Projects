from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf


def weight_variable(shape, stddev=0.1, name='weight'):
    initial = tf.random.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)

def weight_variable_devonc(shape, stddev=0.1, name='weight_devonc'):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=stddev), name=name)

def conv2d(x, W, b, keep_prob_):
    with tf.name_scope('conv2d'):
        conv