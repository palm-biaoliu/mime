# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
################################################################################

"""

Authors: liubiao(liubiao04@baidu.com)
Date:    2021/11/06 16:52
"""
import logging
import sys
import os
import time

_ALL_LOGGERS = {}


def get_logger(name=None, dir=None, logFileName=None):
    logger = logging.getLogger(name)
    if name in _ALL_LOGGERS:
        return logger
    # fmt = "%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] %(thread)d: %(message)s"
    fmt = "%(asctime)s : %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(formatter)
    std_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        os.path.join(dir, logFileName), encoding='utf8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.handlers = []
    # logger.addHandler(std_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.DEBUG)
    _ALL_LOGGERS[name] = logger
    return logger


def initLogger(args, save_dir='results/'):
    id = str(time.time())
    save_dir = save_dir + str(args.loss) + '/' + '/'.join([str(args.dataset)]) + '/' + id
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(save_dir + '/result_data')
    logfilename = '_'.join([str(args.dataset)]) + '_' + id + '.log'
    logger = get_logger('results_log', save_dir, logfilename)
    logger.info('\nParameters :\n' + '\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    return logger, save_dir


def get_logger_by_name(name='results_log'):
    return _ALL_LOGGERS[name]
