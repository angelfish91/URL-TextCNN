#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-
"""
测试模块

"""
import os
import json

import numpy as np
import tensorflow as tf

from utils import get_word_vocab, ngram_id_x_from_dict, get_words, get_char_id_x, pad_seq_in_word, pad_seq, \
    softmax, batch_iter
from config import MAX_LENGTH_WORDS, MAX_LENGTH_CHARS, MAX_LENGTH_SUBWORDS, NGRAMS_DICT_FILE, WORDS_DICT_FILE, \
    CHARS_DICT_FILE, OUTPUT_DIR, CHECKPOINT_DIR, DELIMIT_MODE, EMB_DIM, BATCH_SIZE_TEST


class DataPreprocessTest(object):
    """
    准备测试数据
    """
    def __init__(self):
        self.chars_dict = None
        self.ngrams_dict = None
        self.words_dict = None
        self.chared_id_x = None
        self.ngramed_id_x = None
        self.worded_id_x = None

    def load_dict(self, file_name):
        """
        加载字典
        :param file_name:
        :return:
        """
        dict_data = None
        file_path = os.path.join(OUTPUT_DIR, file_name)
        if os.path.isfile(file_path):
            with open(file_path, 'r') as fd:
                dict_data = fd.read()
                dict_data = json.loads(dict_data.strip())
        return dict_data

    def do_preprocess(self, url_list):
        """
        测试数据预处理
        :param url_list:
        :return:
        """
        self.chars_dict = self.load_dict(CHARS_DICT_FILE)
        self.ngrams_dict = self.load_dict(NGRAMS_DICT_FILE)
        self.words_dict = self.load_dict(WORDS_DICT_FILE)

        x, word_reverse_dict = get_word_vocab(url_list, MAX_LENGTH_WORDS)
        word_x = get_words(x, word_reverse_dict, DELIMIT_MODE, url_list)

        self.ngramed_id_x, self.worded_id_x = \
            ngram_id_x_from_dict(word_x, MAX_LENGTH_SUBWORDS, self.ngrams_dict, self.words_dict)
        self.chared_id_x = get_char_id_x(url_list, self.chars_dict, MAX_LENGTH_CHARS)


class UrlCheck(object):
    """
    根据训练结果测试数据
    """
    def __init__(self):
        self.sess = None
        self.checkpoint_file = None
        self.input_x_char_seq = None
        self.input_x_word = None
        self.input_x_char = None
        self.input_x_char_pad_idx = None
        self.dropout_keep_prob = None
        self.predictions = None
        self.scores = None

    def load_model(self):
        """
        加载模型文件
        :return:
        """
        self.checkpoint_file = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                saver.restore(self.sess, self.checkpoint_file)

                self.input_x_char_seq = graph.get_operation_by_name("input_x_char_seq").outputs[0]
                self.input_x_word = graph.get_operation_by_name("input_x_word").outputs[0]
                self.input_x_char = graph.get_operation_by_name("input_x_char").outputs[0]
                self.input_x_char_pad_idx = graph.get_operation_by_name("input_x_char_pad_idx").outputs[0]
                self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                self.scores = graph.get_operation_by_name("output/scores").outputs[0]

    def test_step(self, x_tst):
        drop_out_prob = 1.0
        feed_dict = {
                self.input_x_char_seq: x_tst[0],
                self.input_x_word: x_tst[1],
                self.input_x_char: x_tst[2],
                self.input_x_char_pad_idx: x_tst[3],
                self.dropout_keep_prob: drop_out_prob}
        preds, preds_prob = self.sess.run([self.predictions, self.scores], feed_dict)
        return preds, preds_prob

    def do_predict(self, urls):
        """
        进行测试的主函数
        :param urls:
        :return:
        """
        # 加载模型文件，测试数据预处理

        data_obj = DataPreprocessTest()
        data_obj.do_preprocess(urls)
        batches = batch_iter(list(zip(data_obj.ngramed_id_x,
                                      data_obj.worded_id_x,
                                      data_obj.chared_id_x)),
                             BATCH_SIZE_TEST, num_epochs=1, shuffle=False)
        all_predictions = []
        all_scores = []

        for index, batch in enumerate(batches):
            if index % 1000 == 0:
                print("Processing #batch {}".format(index))

            x_char, x_word, x_char_seq = zip(*batch)
            x_batch = []
            x_char_seq = pad_seq_in_word(x_char_seq, MAX_LENGTH_CHARS)
            x_batch.append(x_char_seq)
            x_word = pad_seq_in_word(x_word, MAX_LENGTH_WORDS)
            x_batch.append(x_word)
            x_char, x_char_pad_idx = pad_seq(x_char, MAX_LENGTH_WORDS, MAX_LENGTH_SUBWORDS, EMB_DIM)
            x_batch.extend([x_char, x_char_pad_idx])

            batch_predictions, batch_scores = self.test_step(x_batch)
            all_predictions = np.concatenate([all_predictions, batch_predictions]) 
            all_scores.extend(batch_scores)
        softmax_scores = [softmax(score) for score in all_scores]
        return all_predictions, softmax_scores
