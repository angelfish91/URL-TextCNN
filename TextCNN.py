#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-
"""
网络结构定义

"""
import tensorflow as tf


class TextCNN(object):
    def __init__(self, char_ngram_vocab_size, word_ngram_vocab_size, char_vocab_size, word_seq_len, char_seq_len,
            embedding_size, l2_reg_lambda, filter_sizes):

        self.input_x_char = tf.placeholder(tf.int32, [None, None, None], name="input_x_char")
        self.input_x_char_pad_idx = \
                tf.placeholder(tf.float32, [None, None, None, embedding_size], name="input_x_char_pad_idx")
        self.input_x_word = tf.placeholder(tf.int32, [None, None], name="input_x_word")
        self.input_x_char_seq = tf.placeholder(tf.int32, [None, None], name="input_x_char_seq")

        self.input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        l2_loss = tf.constant(0.0)
        with tf.name_scope("embedding"):
            self.char_w = tf.Variable(tf.random_uniform(
                    [char_ngram_vocab_size, embedding_size], -1.0, 1.0), name="char_emb_w")
            self.word_w = tf.Variable(tf.random_uniform(
                    [word_ngram_vocab_size, embedding_size], -1.0, 1.0), name="word_emb_w")
            self.char_seq_w = tf.Variable(tf.random_uniform(
                    [char_vocab_size, embedding_size], -1.0, 1.0), name="char_seq_emb_w")

            self.embedded_x_char = tf.nn.embedding_lookup(self.char_w, self.input_x_char)
            self.embedded_x_char = tf.multiply(self.embedded_x_char, self.input_x_char_pad_idx)
            self.embedded_x_word = tf.nn.embedding_lookup(self.word_w, self.input_x_word)
            self.embedded_x_char_seq = tf.nn.embedding_lookup(self.char_seq_w, self.input_x_char_seq)

            self.sum_ngram_x_char = tf.reduce_sum(self.embedded_x_char, 2)
            self.sum_ngram_x = tf.add(self.sum_ngram_x_char, self.embedded_x_word)

            self.sum_ngram_x_expanded = tf.expand_dims(self.sum_ngram_x, -1)
            self.char_x_expanded = tf.expand_dims(self.embedded_x_char_seq, -1)

        #  WORD CONVOLUTION LAYER
        pooled_x = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv_maxpool_%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, 256]
                b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                conv = tf.nn.conv2d(self.sum_ngram_x_expanded, w, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, word_seq_len - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding="VALID", name="pool")
                pooled_x.append(pooled)

        num_filters_total = 256 * len(filter_sizes)
        self.h_pool = tf.concat(pooled_x, 3)
        self.x_flat = tf.reshape(self.h_pool, [-1, num_filters_total], name="pooled_x")
        self.h_drop = tf.nn.dropout(self.x_flat, self.dropout_keep_prob, name="dropout_x")

        # CHAR CONVOLUTION LAYER
        pooled_char_x = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("char_conv_maxpool_%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, 256]
                b = tf.Variable(tf.constant(0.1, shape=[256]), name="b")
                w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="w")
                conv = tf.nn.conv2d(self.char_x_expanded, w, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, char_seq_len - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding="VALID", name="pool")
                pooled_char_x.append(pooled)
        num_filters_total = 256 * len(filter_sizes)
        self.h_char_pool = tf.concat(pooled_char_x, 3)
        self.char_x_flat = tf.reshape(self.h_char_pool, [-1, num_filters_total], name="pooled_char_x")
        self.char_h_drop = tf.nn.dropout(self.char_x_flat, self.dropout_keep_prob, name="dropout_char_x")

        with tf.name_scope("word_char_concat"):
            ww = tf.get_variable(
                "ww", shape=(num_filters_total, 512), initializer=tf.contrib.layers.xavier_initializer())
            bw = tf.Variable(tf.constant(0.1, shape=[512]), name="bw")
            l2_loss += tf.nn.l2_loss(ww)
            l2_loss += tf.nn.l2_loss(bw)
            word_output = tf.nn.xw_plus_b(self.h_drop, ww, bw)

            wc = tf.get_variable(
                "wc", shape=(num_filters_total, 512), initializer=tf.contrib.layers.xavier_initializer())
            bc = tf.Variable(tf.constant(0.1, shape=[512]), name="bc")
            l2_loss += tf.nn.l2_loss(wc)
            l2_loss += tf.nn.l2_loss(bc)
            char_output = tf.nn.xw_plus_b(self.char_h_drop, wc, bc)
            self.conv_output = tf.concat([word_output, char_output], 1)

        with tf.name_scope("output"):
            w0 = tf.get_variable("w0", shape=[1024, 512], initializer=tf.contrib.layers.xavier_initializer())
            b0 = tf.Variable(tf.constant(0.1, shape=[512]), name="b0")
            l2_loss += tf.nn.l2_loss(w0)
            l2_loss += tf.nn.l2_loss(b0)
            output0 = tf.nn.relu(tf.matmul(self.conv_output, w0) + b0)

            w1 = tf.get_variable("w1", shape=[512, 256], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[256]), name="b1")
            l2_loss += tf.nn.l2_loss(w1)
            l2_loss += tf.nn.l2_loss(b1)
            output1 = tf.nn.relu(tf.matmul(output0, w1) + b1)

            w2 = tf.get_variable("w2", shape=[256, 128], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[128]), name="b2")
            l2_loss += tf.nn.l2_loss(w2)
            l2_loss += tf.nn.l2_loss(b2)
            output2 = tf.nn.relu(tf.matmul(output1, w2) + b2)

            w = tf.get_variable("w", shape=(128, 2), initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(w)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(output2, w, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            correct_preds = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, "float"), name="accuracy")
