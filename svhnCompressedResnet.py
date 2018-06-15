#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import argparse
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('../../../tensorpack')
from tensorpack import *
#from tensorpack import InputDesc,
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from compressModel import read_cfg
from cifarCompressedResnet import Model

import tensorflow as tf
from tensorflow.contrib.layers import variance_scaling_initializer

BATCH_SIZE = 128
NUM_UNITS = None
OUTDIR = ''
NUM_CLASS = 10

structure = []
discard_first_block = []

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

def get_config():
    logger.auto_set_dir(outdir=OUTDIR)
    dataset_train = get_data('train')
    dataset_test = get_data('test')
    MAX_EPOCH = 200
    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test, 
                [ClassificationError()]),
            ScheduledHyperParamSetter('learning_rate',
                [(1, 0.1), (20, 0.01), (28, 0.001), (50, 0.0001)])
        ],
        model=Model(NUM_CLASS, structure, discard_first_block, n=NUM_UNITS),
        max_epoch = MAX_EPOCH,
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

    args = parser.parse_args()
    NUM_UNITS = args.num_units
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.output:
        OUTDIR = "." + args.output
    if args.cfg:
        NUM_UNITS, structure, discard_first_block, model_path = read_cfg(args.cfg)
        structure = np.add(structure, discard_first_block)
        print(model_path)
    else:
        structure = [NUM_UNITS] * 3
        discard_first_block = [0] * 3
    config = get_config()
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    if args.cfg:
        eval_on_cifar(model_path)
        sys.exit()
    SyncMultiGPUTrainer(config).train()
