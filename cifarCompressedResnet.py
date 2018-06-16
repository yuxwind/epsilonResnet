#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import argparse
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('../tensorpack')
from tensorpack import *
#from tensorpack import InputDesc,
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from compressModel import read_cfg

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer
#import pdb
"""
CIFAR10 ResNet example. See:
Deep Residual Learning for Image Recognition, arxiv:1512.03385
This implementation uses the variants proposed in:
Identity Mappings in Deep Residual Networks, arxiv:1603.05027

I can reproduce the results on 2 TitanX for
n=5, about 7.1% val error after 67k steps (20.4 step/s)
n=18, about 5.95% val error after 80k steps (5.6 step/s, not converged)
n=30: a 182-layer network, about 5.6% val error after 51k steps (3.4 step/s)
This model uses the whole training set instead of a train-val split.

To train:
    ./cifar10-resnet.py --gpu 0,1
"""

BATCH_SIZE = 128
NUM_UNITS = None
OUTDIR = ''
IS_CIFAR10 = True
NUM_CLASS = 10

structure = []
discard_first_block = []

class Model(ModelDesc):

    def __init__(self, NUM_CLASS, structure, discard_first_block, n):
        super(Model, self).__init__()
        self.n = n
        self.NUM_CLASS = NUM_CLASS
        self.structure = structure
        self.discard_first_block = discard_first_block

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 128.0
        assert tf.test.is_gpu_available()
        image = tf.transpose(image, [0, 3, 1, 2])

        cnt = tf.placeholder(tf.int32, [None,784], name='x-input')
        
        def first_block(name, l):
            in_channel = l.get_shape().as_list()[1]
            out_channel = in_channel * 2
            
            grp = int(name[3])
            if self.discard_first_block[grp-1] == 1:
                l = AvgPooling('pool', l, 2)
                l = tf.pad(l, [[0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])
            else:
                l = residual(name, l, increase_dim=True)
            return l
            

        def residual(name, l, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[1]

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1
            
            #implement: full pre-activation
            with tf.variable_scope(name) as scope:
                b1 = l if first else BNReLU(l)
                c1 = Conv2D('conv1', b1, out_channel, stride=stride1, nl=BNReLU)
                c2 = Conv2D('conv2', c1, out_channel)
                if increase_dim:
                    l = AvgPooling('pool', l, 2)
                    l = tf.pad(l, [[0, 0], [in_channel // 2, in_channel // 2], [0, 0], [0, 0]])

                l = c2 + l
                return l
            l = AvgPooling('pool', l, 2)
        with argscope([Conv2D, AvgPooling, BatchNorm, GlobalAvgPooling], data_format='NCHW'), \
                argscope(Conv2D, nl=tf.identity, use_bias=False, kernel_shape=3,
                         W_init=variance_scaling_initializer(mode='FAN_OUT')):
            #pdb.set_trace()
            l = Conv2D('conv0', image, 16, nl=BNReLU)
            l = residual('res1.0', l, first=True)
            for k in range(1, self.structure[0]):
                l = residual('res1.{}'.format(k), l)
            # 32,c=16

            #l = residual('res2.0', l, increase_dim=True)
            l = first_block('res2.0', l)
            for k in range(1, self.structure[1]):
                l = residual('res2.{}'.format(k), l)
            # 16,c=32

            #l = residual('res3.0', l, increase_dim=True)
            l = first_block('res3.0', l)
            for k in range(1, self.structure[2]):
                l = residual('res3.' + str(k), l)
            l = BNReLU('bnlast', l)
            # 8,c=64
            l = GlobalAvgPooling('gap', l)
            
            logits = FullyConnected('linear', l, out_dim=self.NUM_CLASS, nl=tf.identity)
        
        print("logits: "+str(logits.shape))
        #prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        print("cost1: "+str(cost.shape))
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        print("cost2: "+str(cost.shape))

        wrong = prediction_incorrect(logits, label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W of fc layers
        wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                          480000, 0.2, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.01, summary=True)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    print("=================1 cifar10 = %r" % IS_CIFAR10)
    if IS_CIFAR10:
        ds = dataset.Cifar10(train_or_test)
    else:
        ds = dataset.Cifar100(train_or_test)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds


def get_config():
    logger.set_logger_dir('train_log.' + out_dir)
    dataset_train = get_data('train')
    dataset_test = get_data('test')
    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                [ClassificationError()]),
            ScheduledHyperParamSetter('learning_rate',
                [(1, 0.1), (82, 0.01), (123, 0.001), (300, 0.0002)])
        ],
        model=Model(NUM_CLASS, structure, discard_first_block, n=NUM_UNITS),
        max_epoch = 10,
        #max_epoch=1,
    )

def eval_on_cifar(model_file):
    print('structure: {}'.format(structure))
    ds = get_data('test')
    pred_config = PredictConfig(
        model=Model(
            NUM_CLASS, structure, discard_first_block, NUM_UNITS),
        session_init=get_model_loader(model_file),
        input_names = ['input', 'label'],
        output_names = ['incorrect_vector']
    )
    pred = SimpleDatasetPredictor(pred_config, ds)
    acc = RatioCounter()
    for o in pred.get_result():
        batch_size = o[0].shape[0]
        acc.feed(o[0].sum(), batch_size)
    print("Error: {}".format(acc.ratio))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', required = True)
    parser.add_argument('-n', '--num_units',
            help='number of units in each stage',
            type=int, default=18)
    parser.add_argument('-o', '--output', help='output', type=str)
    parser.add_argument('--cfg', help = 'config of compressed model', required = True)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--cifar10', help='iscifar10', dest= 'dataset',action = 'store_true')
    feature_parser.add_argument('--cifar100', help='iscifar100', dest= 'dataset',action = 'store_false')

    parser.set_defaults(feature=True)
    args = parser.parse_args()
    NUM_UNITS = args.num_units 
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.output:
        OUTDIR = "." + args.output
    IS_CIFAR10 = args.dataset
    if not IS_CIFAR10:
        NUM_CLASS  = 100
    print('is_cifar10 %r' % IS_CIFAR10)
    if args.cfg:
        NUM_UNITS, structure, discard_first_block, model_path = read_cfg(args.cfg)
        structure = np.add(structure, discard_first_block)
        print(model_path)
    else:
        structure = [NUM_UNITS] * 3
        discard_first_block = [0] * 3
    config = get_config()
    #pdb.set_trace()    
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    if args.cfg:
        eval_on_cifar(model_path)
        sys.exit()
    SyncMultiGPUTrainer(config).train()
    #SimpleTrainer(config).train()
