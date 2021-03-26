# -*- coding: utf-8 -*-
# @Time : 2021/03/19 10:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : word2vec_textrnn.py
# @Software: PyCharm
from abc import ABC
import tensorflow as tf
from config import classifier_config


class TextRNN(tf.keras.Model, ABC):
    """
    TextRNN模型
    """

    def __init__(self, seq_length, num_classes, hidden_dim, embedding_dim, vocab_size):
        super(TextRNN, self).__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.embedding_method = classifier_config['embedding_method']
        self.embedding = tf.keras.layers.Embedding(vocab_size + 1, self.embedding_dim, mask_zero=True)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.hidden_dim, return_sequences=True))
        self.dropout = tf.keras.layers.Dropout(classifier_config['dropout_rate'], name='dropout')

        if classifier_config['use_attention']:
            attention_size = classifier_config['attention_size']
            self.attention_W = tf.keras.layers.Dense(attention_size, activation='tanh', use_bias=True)
            self.attention_V = tf.keras.layers.Dense(1)

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
        bilstm_outputs = self.bilstm(inputs)

        if classifier_config['use_attention']:
            u_list = []
            attn_z = []
            attention_inputs = tf.split(tf.reshape(inputs, [-1, self.embedding_dim]), self.seq_length, axis=0)
            for t in range(self.seq_length):
                u_t = self.attention_W(attention_inputs[t])
                u_list.append(u_t)

            for t in range(self.seq_length):
                z_t = self.attention_V(u_list[t])
                attn_z.append(z_t)

            attn = tf.concat(attn_z, axis=1)
            alpha = tf.nn.softmax(attn)
            alpha = tf.reshape(alpha, [-1, self.seq_length, 1])
            bilstm_outputs = alpha * bilstm_outputs

        dropout_outputs = self.dropout(bilstm_outputs, training)
        flatten_outputs = self.flatten(dropout_outputs)
        outputs = self.dense(flatten_outputs)
        return outputs
