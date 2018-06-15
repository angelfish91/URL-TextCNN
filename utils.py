#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-
"""
辅助函数

"""

import numpy as np

from tensorflow.contrib import learn
from tflearn.data_utils import to_categorical

from config import logger, module_name


def read_data(input_path):
    """
    读取训练数据
    :param input_path:
    :return:
    """
    with open(input_path, "r") as fp:
        urls, labels = list(), list()
        for line in fp.readlines():
            try:
                label, url = line.split('\t', 1)
                if int(label) == 1:
                    labels.append(1)
                else:
                    labels.append(0)
                urls.append(url)
            except Exception as err:
                logger.error("%s: utils: read_data: data fmt error %s %s" % (module_name, line, err))
    return urls, labels 


def get_word_vocab(urls, max_length_words, min_word_freq=0):
    """
    获取词汇表
    :param urls:
    :param max_length_words:
    :param min_word_freq:
    :return:
    """
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_length_words, min_frequency=min_word_freq)
    x_train = np.array(list(vocab_processor.fit_transform(urls)))
    vocab_dict = vocab_processor.vocabulary_._mapping
    reverse_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    logger.info("%s: Size of word vocabulary: %d" % (module_name, len(vocab_dict)))
    return x_train, reverse_dict


def get_words(x_train, reverse_dict, delimit_mode, urls=None):
    """
    若delimit == 0， 获得URL分词词汇
    若delimit == 1， 获得URL分词词汇 + 特殊字符（非分词词汇）
    :param x_train:
    :param reverse_dict:
    :param delimit_mode:
    :param urls:
    :return:
    """
    processed_x = list()
    if delimit_mode == 0: 
        for url in x_train:
            words = list()
            for word_id in url: 
                if word_id != 0: 
                    words.append(reverse_dict[word_id])
                else: 
                    break
            processed_x.append(words) 
    elif delimit_mode == 1: 
        for x_index in range(x_train.shape[0]):
            word_url = x_train[x_index]
            raw_url = urls[x_index]
            words = list()
            for word_index in range(len(word_url)):
                word_id = word_url[word_index]
                if word_id == 0: 
                    words.extend(list(raw_url))
                    break
                else: 
                    word = reverse_dict[word_id]
                    idx = raw_url.index(word) 
                    special_chars = list(raw_url[0:idx])
                    words.extend(special_chars) 
                    words.append(word) 
                    raw_url = raw_url[idx+len(word):]
                    if word_index == len(word_url) - 1:
                        words.extend(list(raw_url))
            processed_x.append(words)
    return processed_x 


def get_char_ngrams(ngram_len, word):
    """
    获取word在char层面的ngram词汇表
    :param ngram_len:
    :param word:
    :return:
    """
    word = "".join(["<", word, ">"])
    chars = list(word) 
    begin_idx = 0
    ngrams = list()
    while (begin_idx + ngram_len) <= len(chars): 
        end_idx = begin_idx + ngram_len 
        ngrams.append("".join(chars[begin_idx:end_idx])) 
        begin_idx += 1 
    return ngrams 


def get_char_id_x(urls, char_dict, max_len_chars):
    """
    获取url字符映射id列表
    :param urls:
    :param char_dict:
    :param max_len_chars:
    :return:
    """
    char_id_x = list()
    for url in urls: 
        url = list(url) 
        url_in_char_id = list()
        process_length = min(len(url), max_len_chars)
        for char_index in range(process_length):
            char = url[char_index]
            char_id = 0
            if char in char_dict:
                char_id = char_dict[char]
            url_in_char_id.append(char_id)
        char_id_x.append(url_in_char_id)
    return char_id_x


def ngram_id_x(word_x, max_len_subwords, high_freq_words=None):
    """
    获取 ngramed_id_x， worded_id_x 及对应的映射字典
    :param word_x:
    :param max_len_subwords:
    :param high_freq_words:
    :return:
    """
    char_ngram_len = 1
    all_ngrams = set() 
    ngramed_x = []
    all_words = set() 
    worded_x = []

    for index, word_list in enumerate(word_x):
        if index % 100000 == 0:
            print("Processing #url {}".format(index))

        url_in_ngrams = []
        url_in_words = []
        for word in word_list:
            ngrams = get_char_ngrams(char_ngram_len, word) 
            if (len(ngrams) > max_len_subwords) or \
                    (high_freq_words is not None and len(word) > 1 and word not in high_freq_words):
                all_ngrams.update(ngrams[:max_len_subwords])
                url_in_ngrams.append(ngrams[:max_len_subwords]) 
                all_words.add("<UNKNOWN>")
                url_in_words.append("<UNKNOWN>")
            else:     
                all_ngrams.update(ngrams)
                url_in_ngrams.append(ngrams) 
                all_words.add(word) 
                url_in_words.append(word) 
        ngramed_x.append(url_in_ngrams)
        worded_x.append(url_in_words) 

    all_ngrams = list(all_ngrams) 
    ngrams_dict = dict()
    for index in range(len(all_ngrams)):
        ngrams_dict[all_ngrams[index]] = index + 1
    print("Size of ngram vocabulary: {}".format(len(ngrams_dict))) 

    all_words = list(all_words)
    words_dict = dict() 
    for index in range(len(all_words)):
        words_dict[all_words[index]] = index + 1
    print("Size of word vocabulary: {}".format(len(words_dict)))
    print("Index of <UNKNOWN> word: {}".format(words_dict.get("<UNKNOWN>", 0)))

    ngramed_id_x = []
    for ngramed_url in ngramed_x: 
        url_in_ngrams = []
        for ngramed_word in ngramed_url: 
            ngram_ids = [ngrams_dict[x] for x in ngramed_word] 
            url_in_ngrams.append(ngram_ids) 
        ngramed_id_x.append(url_in_ngrams)  

    worded_id_x = []
    for worded_url in worded_x: 
        word_ids = [words_dict[x] for x in worded_url]
        worded_id_x.append(word_ids) 

    return ngramed_id_x, ngrams_dict, worded_id_x, words_dict


def ngram_id_x_from_dict(word_x, max_len_subwords, ngram_dict, word_dict):
    """
    从给定的字典中获取ngram序列，及单词序列的向量表达
    :param word_x:
    :param max_len_subwords:
    :param ngram_dict:
    :param word_dict:
    :return:
    """
    char_ngram_len = 1
    print("Index of <UNKNOWN> word: {}".format(word_dict.get("<UNKNOWN>", 0)))
    ngramed_id_x = [] 
    worded_id_x = []

    word_vocab = set(sorted(list(word_dict.keys())))

    for counter, url in enumerate(word_x):
        if counter % 100000 == 0: 
            print("Processing url #{}".format(counter))
        url_in_ngrams = [] 
        url_in_words = [] 
        words = url
        for word in words:
            ngrams = get_char_ngrams(char_ngram_len, word) 
            if len(ngrams) > max_len_subwords:
                word = "<UNKNOWN>"  
            ngrams_id = [] 
            for ngram in ngrams: 
                if ngram in ngram_dict: 
                    ngrams_id.append(ngram_dict[ngram]) 
                else: 
                    ngrams_id.append(0) 
            url_in_ngrams.append(ngrams_id)
            if word in word_vocab:
                word_id = word_dict[word]
            else: 
                word_id = word_dict["<UNKNOWN>"] 
            url_in_words.append(word_id)
        ngramed_id_x.append(url_in_ngrams)
        worded_id_x.append(url_in_words)
    
    return ngramed_id_x, worded_id_x 


def prep_train_test(pos_x, neg_x, dev_ratio):
    """
    构建训练测试集
    :param pos_x:
    :param neg_x:
    :param dev_ratio: 测试集比例
    :return:
    """
    np.random.seed(10) 
    shuffle_indices = np.random.permutation(np.arange(len(pos_x)))
    pos_x_shuffled = pos_x[shuffle_indices]
    dev_idx = -1 * int(dev_ratio * float(len(pos_x)))
    pos_train = pos_x_shuffled[:dev_idx]
    pos_test = pos_x_shuffled[dev_idx:]

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(neg_x)))
    neg_x_shuffled = neg_x[shuffle_indices]
    dev_idx = -1 * int(dev_ratio * float(len(neg_x)))
    neg_train = neg_x_shuffled[:dev_idx]
    neg_test = neg_x_shuffled[dev_idx:] 

    x_train = np.array(list(pos_train) + list(neg_train))
    y_train = len(pos_train)*[1] + len(neg_train)*[0]
    x_test = np.array(list(pos_test) + list(neg_test))
    y_test = len(pos_test)*[1] + len(neg_test)*[0]
    y_train = to_categorical(y_train, nb_classes=2)
    y_test = to_categorical(y_test, nb_classes=2) 

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    np.random.seed(10) 
    shuffle_indices = np.random.permutation(np.arange(len(x_test)))
    x_test = x_test[shuffle_indices]
    y_test = y_test[shuffle_indices] 
    
    print("Train Mal/Ben split: {}/{}".format(len(pos_train), len(neg_train)))
    print("Test Mal/Ben split: {}/{}".format(len(pos_test), len(neg_test)))
    print("Train/Test split: {}/{}".format(len(y_train), len(y_test)))
    print("Train/Test split: {}/{}".format(len(x_train), len(x_test)))

    return x_train, y_train, x_test, y_test 


def get_ngramed_id_x(x_idxs, ngramed_id_x):
    """
    获取指定index列表的 ngramed_id_x
    :param x_idxs:
    :param ngramed_id_x:
    :return:
    """
    output_ngramed_id_x = list()
    for idx in x_idxs:  
        output_ngramed_id_x.append(ngramed_id_x[idx])
    return output_ngramed_id_x


def pad_seq(urls, max_d1=0, max_d2=0, embedding_size=128):
    """
    对2维序列进行标准化
    :param urls:
    :param max_d1:
    :param max_d2:
    :param embedding_size:
    :return:
    """
    if max_d1 == 0 and max_d2 == 0:
        for url in urls:
            if len(url) > max_d1: 
                max_d1 = len(url) 
            for word in url:
                if len(word) > max_d2: 
                    max_d2 = len(word) 
    pad_idx = np.zeros((len(urls), max_d1, max_d2, embedding_size))
    pad_urls = np.zeros((len(urls), max_d1, max_d2))
    pad_vec = [1] * embedding_size
    for d0 in range(len(urls)): 
        url = urls[d0]
        for d1 in range(len(url)): 
            if d1 < max_d1: 
                word = url[d1]
                for d2 in range(len(word)): 
                    if d2 < max_d2: 
                        pad_urls[d0, d1, d2] = word[d2]
                        pad_idx[d0, d1, d2] = pad_vec
    return pad_urls, pad_idx


def pad_seq_in_word(urls, max_d1=0):
    """
    对词序列1维向量进行标准化向量长度
    :param urls:
    :param max_d1:
    :return:
    """
    if max_d1 == 0: 
        url_lens = [len(url) for url in urls]
        max_d1 = max(url_lens)
    pad_urls = np.zeros((len(urls), max_d1))
    for d0 in range(len(urls)):
        url = urls[d0]
        for d1 in range(len(url)): 
            if d1 < max_d1: 
                pad_urls[d0, d1] = url[d1]
    return pad_urls


def softmax(value):
    """
    计算softmax值，将神经网络结果归一化
    :param value:
    :return:
    """
    exp_value = np.exp(value - np.max(value))
    return exp_value / exp_value.sum()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    为网络训练数据输入建立迭代器
    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    data = np.array(data) 
    data_size = len(data) 
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1 
    for epoch in range(num_epochs): 
        if shuffle: 
            shuffle_indices = np.random.permutation(np.arange(data_size)) 
            shuffled_data = data[shuffle_indices]
        else: 
            shuffled_data = data 
        for batch_num in range(num_batches_per_epoch): 
            start_idx = batch_num * batch_size 
            end_idx = min((batch_num+1) * batch_size, data_size)
            yield shuffled_data[start_idx:end_idx]
