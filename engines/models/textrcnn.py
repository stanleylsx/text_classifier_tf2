# -*- coding: utf-8 -*-
# @Time : 2020/11/6 10:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : textrcnn.py
# @Software: PyCharm
from abc import ABC
import tensorflow as tf
from config import classifier_config


class TextRCNN(tf.keras.Model, ABC):
    """
    TextCNN模型
    """
    def __init__(self, seq_length, num_classes, hidden_dim, embedding_dim):
        super(TextRCNN, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))
        # self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[seq_length-2+1, 1], padding='valid')
        self.dropout = tf.keras.layers.Dropout(classifier_config['droupout_rate'], name='dropout')
        self.dense = tf.keras.layers.Dense(2 * self.hidden_dim + self.embedding_dim, activation='tanh')
        self.dense = tf.keras.layers.Dense(num_classes,
                                           activation='softmax',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                           bias_regularizer=tf.keras.regularizers.l2(0.1),
                                           name='dense')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_last', name='flatten')

    # @tf.function
    def call(self, inputs, training=None):
        bilstm_outputs = self.bilstm(inputs)
        concat_outputs = tf.keras.layers.concatenate([bilstm_outputs, inputs], axis=-1, name='concatenate')
        print('asd')
        # fc_input =
        # fc_output = tf.tanh(concat_outputs)
        # flatten_outputs = self.flatten(concat_outputs)
        # dropout_outputs = self.dropout(flatten_outputs, training)
        # outputs = self.dense(dropout_outputs)
        # return outputs
