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
    TextRCNN模型
    """

    def __init__(self, num_classes, hidden_dim, embedding_dim, vocab_size, embeddings_matrix=None):
        super(TextRCNN, self).__init__()
        if classifier_config['embedding_method'] is '':
            self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, mask_zero=True)
        else:
            self.embedding = tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, weights=[embeddings_matrix],
                                                       trainable=False)

        self.forward = tf.keras.layers.LSTM(hidden_dim, return_sequences=True)
        self.backward = tf.keras.layers.LSTM(hidden_dim, return_sequences=True, go_backwards=True)
        self.max_pool = tf.keras.layers.GlobalMaxPool1D()
        self.dropout = tf.keras.layers.Dropout(classifier_config['dropout_rate'], name='dropout')
        self.dense1 = tf.keras.layers.Dense(2 * hidden_dim + embedding_dim, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(num_classes,
                                            activation='softmax',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                            bias_regularizer=tf.keras.regularizers.l2(0.1),
                                            name='dense')

    @tf.function
    def call(self, inputs, training=None):
        inputs = self.embedding(inputs)
        left_embedding = self.forward(inputs)
        right_embedding = self.backward(inputs)
        concat_outputs = tf.keras.layers.concatenate([left_embedding, inputs, right_embedding], axis=-1)
        dropout_outputs = self.dropout(concat_outputs, training)
        fc_outputs = self.dense1(dropout_outputs)
        pool_outputs = self.max_pool(fc_outputs)
        outputs = self.dense2(pool_outputs)
        return outputs
