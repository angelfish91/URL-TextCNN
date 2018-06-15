#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-
"""
配置文件

"""

import os
import logging

logger = logging.getLogger()

# module name
module_name = "url_net_trainning"
# directory o the output model
OUTPUT_DIR = "./runs"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints/")
# dictionary file name
NGRAMS_DICT_FILE = "ngrams_dict.json"
WORDS_DICT_FILE = "words_dict.json"
CHARS_DICT_FILE = "chars_dict.json"

# Max length of url in words
MAX_LENGTH_WORDS = 200
# Max length of url in chars
MAX_LENGTH_CHARS = 200
# Max length of word in ngrams
MAX_LENGTH_SUBWORDS = 20
# Minimum frequency of word to build vocabulary
MIN_WORD_FREQ = 1
# embedding dimension size
EMB_DIM = 32
# umber of training epochs
NB_EPOCHS = 5
# Size of a training batch
BATCH_SIZE_TRAIN = 128
# portion of training used for dev
DEV_PERCENTAGE = 0.005
# print training result every this number of steps
PRINT_EVERY = 50
# evaluate the model every this number of steps
EVAL_EVERY = 500
# Save a model every this number of steps
CHECKPOINT_EVERY = 500
# l2 lambda for regularization
L2_REG_LAMBDA = 0.0
# filter sizes of the convolution layer
FILTER_SIZES = [3, 4, 5, 6]
# learning rate of the optimizer
LR = 0.001
# 0: delimit by special chars, 1: delimit by special chars + each char as a word
DELIMIT_MODE = 1
# Size of a test batch
BATCH_SIZE_TEST = 50