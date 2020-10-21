# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: PyCharm
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from engines.word2vec_util import Word2VecUtils
from config import word2vec_config, textcnn_config


class DataManager:

    def __init__(self, logger):
        self.logger = logger

        self.PADDING = '[PAD]'

        self.train_file = textcnn_config['train_file']
        self.dev_file = textcnn_config['dev_file']
        self.w2v_model = Word2Vec.load(word2vec_config['word2vec_model'])
        self.w2v_util = Word2VecUtils(logger)
        self.stop_words = self.w2v_util.get_stop_words()

        self.batch_size = textcnn_config['batch_size']
        self.max_sequence_length = textcnn_config['max_sequence_length']
        self.embedding_dim = self.w2v_model.vector_size

        self.class_id = textcnn_config['classes']
        self.max_label_number = len(self.class_id)

        self.logger.info('dataManager initialed...')

    def next_batch(self, X, y, start_index):
        """
        下一次个训练批次
        :param X:
        :param y:
        :param start_index:
        :return:
        """
        last_index = start_index + self.batch_size
        X_batch = list(X[start_index:min(last_index, len(X))])
        y_batch = list(y[start_index:min(last_index, len(X))])
        if last_index > len(X):
            left_size = last_index - (len(X))
            for i in range(left_size):
                index = np.random.randint(len(X))
                X_batch.append(X[index])
                y_batch.append(y[index])
        return np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)

    def padding(self, sentence):
        """
        长度不足max_sequence_length则补齐
        :param sentence:
        :return:
        """
        if len(sentence) < self.max_sequence_length:
            sentence += [self.PADDING for _ in range(self.max_sequence_length - len(sentence))]
        else:
            sentence = sentence[:self.max_sequence_length]
        return sentence

    def prepare(self, sentences, labels):
        """
        输出X矩阵和y向量
        """
        self.logger.info('loading data...')
        X, y = [], []
        embedding_unknown = [0] * self.embedding_dim
        for record in tqdm(zip(sentences, labels)):
            sentence = self.w2v_util.processing_sentence(record[0], self.stop_words)
            sentence = self.padding(sentence)
            label = record[1]
            vector = []
            for word in sentence:
                if word in self.w2v_model.wv.vocab:
                    vector.append(self.w2v_model[word].tolist())
                else:
                    vector.append(embedding_unknown)
            X.append(vector)
            y.append(label)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def get_training_set(self, train_val_ratio=0.9):
        """
        获取训练数据集、验证集
        """
        df_train = pd.read_csv(self.train_file)
        df_train['label'] = df_train.label.map(lambda x: self.class_id[x])
        # convert the data in matrix
        X, y = self.prepare(df_train['sentence'], df_train['label'])
        # shuffle the samples
        self.logger.info('shuffling data...')
        num_samples = len(X)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        self.logger.info('getting validation data...')
        if self.dev_file is not None:
            X_train = X
            y_train = y
            X_val, y_val = self.get_valid_set()
        else:
            # split the data into train and validation set
            X_train = X[:int(num_samples * train_val_ratio)]
            y_train = y[:int(num_samples * train_val_ratio)]
            X_val = X[int(num_samples * train_val_ratio):]
            y_val = y[int(num_samples * train_val_ratio):]
            self.logger.info('validating set is not exist, built...')
        self.logger.info('training set size: {}, validating set size: {}'.format(len(X_train), len(y_val)))
        return X_train, y_train, X_val, y_val

    def get_valid_set(self):
        """
        获取验证集
        :return:
        """
        df_val = pd.read_csv(self.dev_file)
        df_val['label'] = df_val.label.map(lambda x: self.class_id[x])
        # convert the data in matrix
        X_val, y_val = self.prepare(df_val['sentence'], df_val['label'])
        return X_val, y_val

    def prepare_single_sentence(self, sentence):
        """
        把预测的句子转成矩阵和向量
        :param sentence:
        :return:
        """
        embedding_unknown = [0] * self.embedding_dim
        sentence = self.w2v_util.processing_sentence(sentence, self.stop_words)
        sentence = self.padding(sentence)
        vector = []
        for word in sentence:
            if word in self.w2v_model.wv.vocab:
                vector.append(self.w2v_model[word].tolist())
            else:
                vector.append(embedding_unknown)
        return np.array([vector], dtype=np.float32)
