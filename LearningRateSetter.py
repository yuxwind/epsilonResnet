#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: LearningRateSetter.py
# Author: Xin Yu<yuxwind@gmail.com>

import tensorflow as tf
from abc import abstractmethod, ABCMeta
import operator
import six
import os

from .base import Callback
from ..utils import logger
from ..tfutils import get_op_tensor_name
from param import HyperParamSetter

__all__ = ['LearningRateSetter']
# put this file under {tensorpack_root}/tensorpack/callbacks
class LearningRateSetter(HyperParamSetter):
    """
    Change learning rate by monitoring the change of a statistic.
    Change when it was increasing enough.
    """
    def __init__(self, param, stat_name, init_schedule, updated_schedule, threshold, last_k):
        """
        Args:
            param: same as in :class:`HyperParamSetter`.
            stat_name (str): name of the statistics.
            init_schedule: inititial schedule for learning rate
            update_schedule: adaptive learning rate policy
            threshold (float): change threshold.
            last_k (int): last k epochs.

        This callback will change plr by ``lr = update_schedule[0]``, when:
        ``min(stats) >= stats[0] + threshold``, where
        ``stats = [stat_name in last k epochs]``

        Example:
            If discarded_cnt was increased in last 1 epoch, follow adaptive learning rate policy:
            .. code-block:: python
            LearningRateSetter('learning_rate','discarded_cnt',
                [(0, 0.1), (82, 0.01), (123, 0.001), (300,0.0002)],
                [(0, 0.1), (41, 0.01), (61, 0.001), (150,0.0002)],
                1,1)
        """
        super(LearningRateSetter, self).__init__(param)
        self.stat_name = stat_name
        self.last_k = last_k
        self.threshold = threshold

        self.last_changed_epoch = 0
        self.updated = False
        init_schedule = [(int(a), float(b)) for a, b in init_schedule]
        self.schedule = sorted(init_schedule, key=operator.itemgetter(0))
        updated_schedule = [(int(a), float(b)) for a, b in updated_schedule]
        self.updated_schedule = sorted(updated_schedule, key=operator.itemgetter(0))

    def _get_value_to_set(self):
        hist = self.trainer.monitors.get_history(self.stat_name)
        cur = self.get_current_value()
        logger.info("[LearningRateSetter] Triggered,%s cur=%f" %(self.param.readable_name, cur))
        if len(hist) < self.last_k + 1 or \
                self.epoch_num - self.last_changed_epoch < self.last_k:
            for e, v in self.schedule:
                if e == self.epoch_num:
                    logger.info("[LearningRateSetter] Triggered, return 1")
                    return v
            return None
        hist = hist[-self.last_k - 1:]    # len==last_k+1
        logger.info("[LearningRateSetter] Triggered, history: " +
                    ','.join(map(str, hist)))

        hist_first = hist[0]
        hist_max = max(hist)
        if hist_max < hist_first + self.threshold:  # not increase enough
            for e, v in self.schedule:
                if e == self.epoch_num:
                    logger.info("[LearningRateSetter] Triggered, return 2")
                    return v
            return None
        logger.info("[LearningRateSetter] Triggered, return 3")
        self.last_changed_epoch = self.epoch_num
        self.schedule =[(int(a)+self.epoch_num, \
                float(b)) for a, b in self.updated_schedule]
        return self.schedule[0][1]
