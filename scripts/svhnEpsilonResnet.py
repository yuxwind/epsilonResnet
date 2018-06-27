#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import argparse
import os
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys

sys.path.append('../../tensorpack')
from tensorpack import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.gradproc import SummaryGradient

from EpsilonResnetBase import *
from compressModel import read_cfg
from cifarEpsilonResnet import Model

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

BATCH_SIZE = 128
EPSILON = 1.5
NUM_UNITS = None
NUM_CLASS = 10

def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    pp_mean = dataset.SVHNDigit.get_per_pixel_mean()
    if isTrain:
        d1 = dataset.SVHNDigit('train')
        d2 = dataset.SVHNDigit('extra')
        ds = RandomMixData([d1, d2])
    else:
        ds = dataset.SVHNDigit('test')

    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.Brightness(10),
            imgaug.Contrast((0.8, 1.2)),
            imgaug.GaussianDeform(  # this is slow. without it, can only reach 1.9% error
                [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
                (40, 40), 0.2, 3),
            imgaug.RandomCrop((32, 32)),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 5, 5)
    return ds

def get_config(out_dir):
    print("outdir: %s"%out_dir)
    logger.set_logger_dir('train_log.' + out_dir)
    dataset_train = get_data('train')
    dataset_test = get_data('test')
    MAX_EPOCH = 200
    side_layers = ['res2.{}'.format(NUM_UNITS/2)]
    side_prediction_name = ["side_output/" +x for x in side_layers]
    side_inferences = [ClassificationError(x+ "/incorrect_vector",\
            x + "/val_error") for x in side_prediction_name]
    inferences = side_inferences + [ScalarStats('cost'), ClassificationError()]
    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test, inferences),
            LearningRateSetter('learning_rate','discarded_cnt',
                [(1, 0.1), (20, 0.01), (28, 0.001), (50, 0.0001)],
                [(1, 0.1), (10, 0.01), (14, 0.001), (25, 0.0001)],
                1,1),
        ],
        model=Model(EPSILON, NUM_CLASS, NUM_UNITS),
        max_epoch = MAX_EPOCH,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=18)
    parser.add_argument('--load', help='load model')
    parser.add_argument('-e', '--epsilon', help='set epsilon')
    parser.add_argument('-o', '--output', help='output')

    args = parser.parse_args()
    NUM_UNITS = args.num_units
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.epsilon:
        EPSILON = float(args.epsilon)
    out_dir = ""
    if args.output:
        out_dir = "." + args.output
    print('epsilon = %f' % EPSILON)
    config = get_config(out_dir)
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()
