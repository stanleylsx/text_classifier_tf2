# -*- coding: utf-8 -*-
# @Time : 2022/06/13 22:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : PretrainedModel.py
# @Software: PyCharm

from abc import ABC
from transformers import TFBertModel
import tensorflow as tf
from config import classifier_config


class PretrainedModelClassification(tf.keras.Model, ABC):
    def __init__(self, num_classes, model_type):
        super(PretrainedModelClassification, self).__init__()
        self.max_sequence_length = classifier_config['max_sequence_length']
        if model_type == 'Bert':
            from transformers import TFBertModel
            self.model = TFBertModel.from_pretrained(classifier_config['pretrained'])
        elif model_type == 'DistilBert':
            from transformers import TFDistilBertModel
            self.model = TFDistilBertModel.from_pretrained(classifier_config['pretrained'])
        elif model_type == 'AlBert':
            from transformers import TFAlbertModel
            self.model = TFAlbertModel.from_pretrained(classifier_config['pretrained'])
        elif model_type == 'RoBerta':
            from transformers import TFRobertaModel
            self.model = TFRobertaModel.from_pretrained(classifier_config['pretrained'])
        elif model_type == 'Electra':
            from transformers import TFElectraModel
            self.model = TFElectraModel.from_pretrained(classifier_config['pretrained'])
        elif model_type == 'XLNet':
            from transformers import TFXLNetModel
            self.model = TFXLNetModel.from_pretrained(classifier_config['pretrained'])
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
        last_hidden_state = self.model(input_ids=inputs, attention_mask=bert_mask_ids).last_hidden_state
        pooled_output = last_hidden_state[:, 0]
        dropout_outputs = self.dropout(pooled_output, training)
        outputs = self.dense(dropout_outputs)
        return outputs
