#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: EpsilonResNetBase.py
# Author: Xin Yu <yuxwind@gmail.com>

import sys
sys.path.append('../../../tensorpack')
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

import tensorflow as tf

"""
Implementation of strict identity and side output in the following paper:
    Learning Strict Identity Mappings in Deep Residual Networks
    (https://arxiv.org/pdf/1804.01661.pdf)
"""
# implement sparsity promting function with 4 ReLUs
#   Usually, l is a 4 dimension tensor: Batch_size X Width X Height X Channel
#   return 0.0 only if the absolute values of all elemenents in l are smaller than EPSILON
def strict_identity(l, EPSILON):
    l = tf.to_float(l)
    s = tf.reduce_max(tf.nn.relu(l - EPSILON) +\
            tf.nn.relu(-l - EPSILON))
    identity_w = tf.nn.relu(tf.nn.relu(s * (-1000000) + 1.0) * (-1000000) + 1.0)
    return identity_w

# implement side supervision at the intermediate of the network
#   get cross entropy loss after layer l
def side_output(name, l, label, outdim):
    prefix = 'side_output/'+name
    with tf.variable_scope(prefix) as scope:
        l = BNReLU('bnlast', l)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, out_dim=outdim, nl=tf.identity)

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        wrong = prediction_incorrect(logits, label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))
    return cost
