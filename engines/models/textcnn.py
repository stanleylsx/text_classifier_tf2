# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : textcnn.py
# @Software: PyCharm
from abc import ABC
import tensorflow as tf
from config import classifier_config


class TextCNN(tf.keras.Model, ABC):
    """
    TextCNN模型
    """
    def __init__(self, seq_length, num_filters, num_classes, embedding_dim, vocab_size):
        super(TextCNN, self).__init__()
        self.seq_length = seq_length
        self.embedding_method = classifier_config['embedding_method']
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size+1, self.embedding_dim, mask_zero=True)

        if classifier_config['use_attention']:
            attention_size = classifier_config['attention_size']
            self.attention_W = tf.keras.layers.Dense(attention_size, activation='tanh', use_bias=True)
            self.attention_V = tf.keras.layers.Dense(1)

        self.conv1 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=[2, self.embedding_dim],
                                            strides=1,
                                            padding='valid',
                                            activation='relu')
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=[seq_length-2+1, 1], padding='valid')

        self.conv2 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=[3, self.embedding_dim], strides=1,
                                            padding='valid',
                                            activation='relu')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=[seq_length-3+1, 1], padding='valid')

        self.conv3 = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=[4, self.embedding_dim], strides=1,
                                            padding='valid',
                                            activation='relu')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=[seq_length-4+1, 1], padding='valid')
        self.dropout = tf.keras.layers.Dropout(classifier_config['dropout_rate'], name='dropout')
        self.dense = tf.keras.layers.Dense(num_classes,
                                           activation='softmax',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.2),
                                           bias_regularizer=tf.keras.regularizers.l2(0.2),
                                           name='dense')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_last', name='flatten')

    @tf.function
    def call(self, inputs, training=None):
        # 不引入外部的embedding
        if self.embedding_method is None:
            inputs = self.embedding(inputs)

        if classifier_config['use_attention']:
            # 此处的attention是直接对embedding层做attention，思路来自于https://kexue.fm/archives/5409#%E6%B3%A8%E6%84%8F%E5%8A%9B
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
            inputs = alpha * inputs

        inputs = tf.expand_dims(inputs, -1)
        pooled_output = []
        con1 = self.conv1(inputs)
        pool1 = self.pool1(con1)
        pooled_output.append(pool1)

        con2 = self.conv2(inputs)
        pool2 = self.pool2(con2)
        pooled_output.append(pool2)

        con3 = self.conv3(inputs)
        pool3 = self.pool3(con3)
        pooled_output.append(pool3)

        concat_outputs = tf.keras.layers.concatenate(pooled_output, axis=-1, name='concatenate')
        flatten_outputs = self.flatten(concat_outputs)
        dropout_outputs = self.dropout(flatten_outputs, training)
        outputs = self.dense(dropout_outputs)
        return outputs
