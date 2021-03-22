# -*- coding: utf-8 -*-
# @Time : 2020/11/6 10:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : textrcnn.py
# @Software: PyCharm
from abc import ABC
import tensorflow as tf
from config import classifier_config


class TextRNN(tf.keras.Model, ABC):
    """
    TextCNN模型
    """

    def __init__(self, seq_length, num_classes, hidden_dim, embedding_dim, vocab_size):
        super(TextRNN, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embedding_method = classifier_config['embedding_method']
        self.embedding = tf.keras.layers.Embedding(vocab_size + 1, self.embedding_dim, mask_zero=True)

        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))
        self.dropout = tf.keras.layers.Dropout(classifier_config['droupout_rate'], name='dropout')
        self.dense = tf.keras.layers.Dense(num_classes,
                                           activation='softmax',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                           bias_regularizer=tf.keras.regularizers.l2(0.1),
                                           name='dense')
        self.flatten = tf.keras.layers.Flatten(name='flatten')

    @tf.function
    def call(self, inputs, training=None):
        # 不引入外部的embedding
        if self.embedding_method is None:
            inputs = self.embedding(inputs)

        dropout_inputs = self.dropout(inputs, training)
        bilstm_outputs = self.bilstm(dropout_inputs)
        flatten_outputs = self.flatten(bilstm_outputs)
        outputs = self.dense(flatten_outputs)
        return outputs
