#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-
"""
数据预处理模块

"""
import os
import json

import numpy as np

from utils import get_word_vocab, get_char_id_x, get_words, prep_train_test, get_ngramed_id_x, ngram_id_x
from config import MAX_LENGTH_CHARS, MAX_LENGTH_WORDS, MAX_LENGTH_SUBWORDS, MIN_WORD_FREQ, DEV_PERCENTAGE, \
    DELIMIT_MODE, OUTPUT_DIR, NGRAMS_DICT_FILE, WORDS_DICT_FILE, CHARS_DICT_FILE


class DataPreprocess(object):
    """
    为神经网络训练提供数据
    """
    def __init__(self):
        self.high_freq_words = None
        self.word_reverse_dict = None
        self.chars_dict = None
        self.ngrams_dict = None
        self.words_dict = None
        self.ngramed_id_x = None
        self.worded_id_x = None
        self.chared_id_x = None
        self.x_train_char = None
        self.x_test_char = None
        self.x_train_word = None
        self.x_test_word = None
        self.x_train_char_seq = None
        self.x_test_char_seq = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def dump_dict(self, data, file_name):
        """
        将训练用的词典映射dump至磁盘
        :param data:
        :param file_name:
        :return:
        """
        if not os.path.isdir(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        dict_file_path = os.path.join(OUTPUT_DIR, file_name)
        try:
            with open(dict_file_path, "w") as fd:
                fd.write(json.dumps(data))
        except Exception as err:
            print("%s dict dump error %s" % (file_name, str(err)))

    def do_preprocess(self, url_list, label_list):
        """
        进行预处理
        :param url_list:
        :param label_list:
        :return:
        """
        if MIN_WORD_FREQ > 0:
            x__, word_reverse_dict = get_word_vocab(url_list, MAX_LENGTH_WORDS, MIN_WORD_FREQ)
            self.high_freq_words = sorted(list(word_reverse_dict.values()))

        self.x, self.word_reverse_dict = get_word_vocab(url_list, MAX_LENGTH_WORDS)
        word_x = get_words(self.x, self.word_reverse_dict, DELIMIT_MODE, url_list)
        self.ngramed_id_x, self.ngrams_dict, self.worded_id_x, self.words_dict = \
            ngram_id_x(word_x, MAX_LENGTH_SUBWORDS, self.high_freq_words)
        self.chars_dict = self.ngrams_dict
        self.chared_id_x = get_char_id_x(url_list, self.chars_dict, MAX_LENGTH_CHARS)

        pos_x, neg_x = list(), list()
        for index in range(len(label_list)):
            label = label_list[index]
            if label == 1:
                pos_x.append(index)
            else:
                neg_x.append(index)
        print("Overall Mal/Ben split: {}/{}".format(len(pos_x), len(neg_x)))
        pos_x = np.array(pos_x)
        neg_x = np.array(neg_x)

        self.x_train, self.y_train, self.x_test, self.y_test = prep_train_test(pos_x, neg_x, DEV_PERCENTAGE)

        self.x_train_char = get_ngramed_id_x(self.x_train, self.ngramed_id_x)
        self.x_test_char = get_ngramed_id_x(self.x_test, self.ngramed_id_x)

        self.x_train_word = get_ngramed_id_x(self.x_train, self.worded_id_x)
        self.x_test_word = get_ngramed_id_x(self.x_test, self.worded_id_x)

        self.x_train_char_seq = get_ngramed_id_x(self.x_train, self.chared_id_x)
        self.x_test_char_seq = get_ngramed_id_x(self.x_test, self.chared_id_x)

        self.dump_dict(self.ngrams_dict, NGRAMS_DICT_FILE)
        self.dump_dict(self.words_dict, WORDS_DICT_FILE)
        self.dump_dict(self.chars_dict, CHARS_DICT_FILE)


