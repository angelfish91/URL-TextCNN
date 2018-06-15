#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-
"""
训练模块

"""
import os

import tensorflow as tf

from TextCNN import TextCNN
from utils import pad_seq, pad_seq_in_word, batch_iter
from config import MAX_LENGTH_CHARS, MAX_LENGTH_WORDS, MAX_LENGTH_SUBWORDS, EMB_DIM, NB_EPOCHS, \
    BATCH_SIZE_TRAIN, CHECKPOINT_DIR, PRINT_EVERY, CHECKPOINT_EVERY, L2_REG_LAMBDA, FILTER_SIZES, LR, EVAL_EVERY


class TrainNet(object):
    """
    训练神经网络
    """
    def __init__(self):
        """
        初始化参数
        """
        self.batches = None
        self.nb_batches = 0
        self.nb_batches_per_epoch = 0
        self.checkpoint_prefix = None
        self.check_checkpoint_dir()

    def check_checkpoint_dir(self):
        """
        准备模型存储文件夹
        :return:
        """
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)
        else:
            files = os.listdir(CHECKPOINT_DIR)
            files = [os.path.join(CHECKPOINT_DIR, _) for _ in files]
            for file in files:
                os.remove(file)
        self.checkpoint_prefix = CHECKPOINT_DIR + "model"
        print ("checkpoint would writing to %s" % CHECKPOINT_DIR)

    def build_bactch(self, data_obj):
        """
        将训练数据划分为各个batch，并计算总的batch数量
        :param data_obj:
        :return:
        """
        batch_data = list(zip(data_obj.x_train_char_seq,
                              data_obj.x_train_word,
                              data_obj.x_train_char,
                              data_obj.y_train))
        self.batches = batch_iter(batch_data, BATCH_SIZE_TRAIN, NB_EPOCHS)

        self.nb_batches_per_epoch = int(len(batch_data) / BATCH_SIZE_TRAIN)
        if len(batch_data) % BATCH_SIZE_TRAIN != 0:
            self.nb_batches_per_epoch += 1

        self.nb_batches = int(self.nb_batches_per_epoch * NB_EPOCHS)
        print("Number of batches in total: {}".format(self.nb_batches))
        print("Number of batches per epoch: {}".format(self.nb_batches_per_epoch))

    def get_x_test(self, data_obj):
        """
        构建验证集 x_test，将data_obj类中测试集数据重整后输出
        [char seq, word, char, char_pad_idx]
        :param data_obj:
        :return:
        """
        x_test = list()
        x_test_char_seq = pad_seq_in_word(data_obj.x_test_char_seq, MAX_LENGTH_CHARS)
        x_test.append(x_test_char_seq)

        x_test_word = pad_seq_in_word(data_obj.x_test_word, MAX_LENGTH_WORDS)
        x_test.append(x_test_word)

        x_test_char, x_test_char_pad_idx = pad_seq(data_obj.x_test_char, MAX_LENGTH_WORDS, MAX_LENGTH_SUBWORDS, EMB_DIM)
        x_test.extend([x_test_char, x_test_char_pad_idx])
        y_test = data_obj.y_test
        return x_test, y_test

    def get_x_train_batch(self, batch):
        """
        将batch中的数据重整后输出
        [char seq, word, char, char_pad_idx]
        :param batch:
        :return:
        """
        x_batch = list()
        x_char_seq, x_word, x_char, y_batch = zip(*batch)

        x_char_seq = pad_seq_in_word(x_char_seq, MAX_LENGTH_CHARS)
        x_batch.append(x_char_seq)

        x_word = pad_seq_in_word(x_word, MAX_LENGTH_WORDS)
        x_batch.append(x_word)

        x_char, x_char_pad_idx = pad_seq(x_char, MAX_LENGTH_WORDS, MAX_LENGTH_SUBWORDS, EMB_DIM)
        x_batch.extend([x_char, x_char_pad_idx])
        return x_batch, y_batch

    def train_dev_step(self, cnn, sess, train_op, global_step, x_train, y_train, is_train=True):
        """
        迭代训练
        :param cnn: 训练网络
        :param sess: tf.sess
        :param train_op:
        :param global_step:
        :param x_train:
        :param y_train:
        :param is_train:
        :return:
        """
        feed_dict = dict()
        if is_train:
            dropout_prob = 0.5
        else:
            dropout_prob = 1.0

        feed_dict[cnn.input_x_char_seq] = x_train[0]
        feed_dict[cnn.input_x_word] = x_train[1]
        feed_dict[cnn.input_x_char] = x_train[2]
        feed_dict[cnn.input_x_char_pad_idx] = x_train[3]
        feed_dict[cnn.input_y] = y_train
        feed_dict[cnn.dropout_keep_prob] = dropout_prob

        if is_train:
            _, step, loss, acc = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
        else:
            step, loss, acc = sess.run([global_step, cnn.loss, cnn.accuracy], feed_dict)
        return step, loss, acc

    def do_train_net(self, data_obj):
        """
        训练网络，主函数入口
        :param data_obj: 数据类
        :return:
        """
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            session_conf.gpu_options.allow_growth = True
            sess = tf.Session(config=session_conf)

            with sess.as_default():
                # 创建网络框架
                cnn = TextCNN(
                        char_ngram_vocab_size=len(data_obj.ngrams_dict) + 1,
                        word_ngram_vocab_size=len(data_obj.words_dict) + 1,
                        char_vocab_size=len(data_obj.chars_dict) + 1,
                        embedding_size=EMB_DIM,
                        word_seq_len=MAX_LENGTH_WORDS,
                        char_seq_len=MAX_LENGTH_CHARS,
                        l2_reg_lambda=L2_REG_LAMBDA,
                        filter_sizes=FILTER_SIZES)
                # 创建全局优化求解器
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(LR)
                train_op = optimizer.apply_gradients(optimizer.compute_gradients(cnn.loss), global_step=global_step)
                # 创建存储类
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
                # 初始化模型
                sess.run(tf.global_variables_initializer())

                x_test, y_test = self.get_x_test(data_obj)
                self.build_bactch(data_obj)
                min_dev_loss = float('Inf')
                # 进行迭代训练
                for idx, batch in enumerate(self.batches):
                    x_batch, y_batch = self.get_x_train_batch(batch)
                    step, loss, acc = self.train_dev_step(
                        cnn, sess, train_op, global_step, x_batch, y_batch, is_train=True)
                    if step % PRINT_EVERY == 0:
                        print("step {}, loss {}, acc {}".format(step, loss, acc))
                    if step % EVAL_EVERY == 0:
                        print("\nEvaluation")
                        step, dev_loss, dev_acc = \
                            self.train_dev_step(cnn, sess, train_op, global_step, x_test, y_test, is_train=False)
                        print("step {}, loss {}, acc {}".format(step, dev_loss, dev_acc))
                        if step % CHECKPOINT_EVERY == 0 or idx == (self.nb_batches - 1):
                            if dev_loss < min_dev_loss:
                                path = saver.save(sess, self.checkpoint_prefix, global_step=step)
                                print("Dev loss improved: {} -> {}".format(min_dev_loss, dev_loss))
                                print("Saved model checkpoint to {}\n".format(path))
                                min_dev_loss = dev_loss
                            else:
                                print("Dev loss did not improve: {} -> {}".format(min_dev_loss, dev_loss))
