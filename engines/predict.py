# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
import tensorflow as tf
from engines.model import TextCNN
from config import textcnn_config


class Predictor:
    def __init__(self, data_manager, logger):
        self.dataManager = data_manager
        seq_length = data_manager.max_sequence_length
        num_classes = data_manager.max_label_number
        embedding_dim = data_manager.embedding_dim
        self.logger = logger
        # 卷集核的个数
        num_filters = textcnn_config['num_filters']
        checkpoints_dir = textcnn_config['checkpoints_dir']
        logger.info('loading model parameter')
        self.text_cnn_model = TextCNN(seq_length, num_filters, num_classes, embedding_dim)
        # 实例化Checkpoint，设置恢复对象为新建立的模型
        checkpoint = tf.train.Checkpoint(model=self.text_cnn_model)
        # 从文件恢复模型参数
        checkpoint.restore(tf.train.latest_checkpoint(checkpoints_dir))
        logger.info('loading model successfully')

    def predict_one(self, sentence):
        """
        对输入的句子分类预测
        :param sentence:
        :return:
        """
        reverse_classes = {class_id: class_name for class_name, class_id in self.dataManager.class_id.items()}
        vector = self.dataManager.prepare_single_sentence(sentence)
        vector = tf.expand_dims(vector, -1)
        logits = self.text_cnn_model.call(inputs=vector)
        prediction = tf.argmax(logits, axis=-1)
        prediction = prediction.numpy()[0]
        return reverse_classes[prediction]
