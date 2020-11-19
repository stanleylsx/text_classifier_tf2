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
    def __init__(self, seq_length, num_filters, num_classes, embedding_dim):
        super(TextCNN, self).__init__()
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim

        if classifier_config['use_attention']:
            self.attention_dim = classifier_config['attention_dim']
            self.attention_W = tf.keras.layers.Dense(classifier_config['attention_dim'], activation='tanh')
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
        self.dropout = tf.keras.layers.Dropout(classifier_config['droupout_rate'], name='dropout')
        self.dense = tf.keras.layers.Dense(num_classes,
                                           activation='softmax',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.1),
                                           bias_regularizer=tf.keras.regularizers.l2(0.1),
                                           name='dense')
        self.flatten = tf.keras.layers.Flatten(data_format='channels_last', name='flatten')

    @tf.function
    def call(self, inputs, training=None):
        if classifier_config['use_attention'] and training:
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
