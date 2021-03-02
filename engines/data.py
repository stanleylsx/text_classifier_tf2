# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from engines.utils.word2vec import Word2VecUtils
from config import classifier_config


class DataManager:

    def __init__(self, logger):
        self.logger = logger

        self.PADDING = '[PAD]'

        self.w2v_util = Word2VecUtils(logger)
        self.w2v_model = Word2Vec.load(self.w2v_util.model_path)

        self.stop_words = self.w2v_util.get_stop_words()

        self.batch_size = classifier_config['batch_size']
        self.max_sequence_length = classifier_config['max_sequence_length']
        self.embedding_dim = self.w2v_model.vector_size

        self.class_id = classifier_config['classes']
        self.class_list = [name for name, index in classifier_config['classes'].items()]
        self.max_label_number = len(self.class_id)

        self.logger.info('dataManager initialed...')

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
            label = tf.one_hot(record[1], depth=self.max_label_number)
            vector = []
            for word in sentence:
                if word in self.w2v_model.wv.vocab:
                    vector.append(self.w2v_model[word].tolist())
                else:
                    vector.append(embedding_unknown)
            X.append(vector)
            y.append(label)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def get_dataset(self, df):
        """
        构建Dataset
        """
        df = df.loc[df.label.isin(self.class_list)]
        df['label'] = df.label.map(lambda x: self.class_id[x])
        # convert the data in matrix
        X, y = self.prepare(df['sentence'], df['label'])
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset

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
