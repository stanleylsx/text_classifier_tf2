# -*- coding: utf-8 -*-
# @Time : 2020/11/03 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : bert.py
# @Software: PyCharm

from abc import ABC
from transformers import TFDistilBertModel
import tensorflow as tf
from config import classifier_config


class DistilBertClassification(tf.keras.Model, ABC):
    def __init__(self, num_classes):
        super(DistilBertClassification, self).__init__()
        self.max_sequence_length = classifier_config['max_sequence_length']
        self.distilbert_model = TFDistilBertModel.from_pretrained(classifier_config['pretrained'])
        self.dropout = tf.keras.layers.Dropout(classifier_config['dropout_rate'], name='dropout')
        self.dense = tf.keras.layers.Dense(num_classes,
                                           activation='softmax',
                                           kernel_regularizer=tf.keras.regularizers.l2(0.2),
                                           bias_regularizer=tf.keras.regularizers.l2(0.2),
                                           name='dense')

    @tf.function
    def call(self, inputs, training=None):
        sequence_len = tf.reduce_sum(tf.sign(inputs), axis=1)
        sequence_len = tf.cast(sequence_len, tf.int32)
        bert_mask_ids = tf.sequence_mask(sequence_len, self.max_sequence_length, tf.int32)
        last_hidden_state = self.distilbert_model(input_ids=inputs, attention_mask=bert_mask_ids).last_hidden_state
        pooled_output = last_hidden_state[:, 0]
        dropout_outputs = self.dropout(pooled_output, training)
        outputs = self.dense(dropout_outputs)
        return outputs
