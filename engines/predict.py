# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
import tensorflow as tf
from config import classifier_config


class Predictor:
    def __init__(self, data_manager, logger):
        hidden_dim = classifier_config['hidden_dim']
        classifier = classifier_config['classifier']
        self.dataManager = data_manager
        seq_length = data_manager.max_sequence_length
        num_classes = data_manager.max_label_number
        embedding_dim = data_manager.embedding_dim
        self.logger = logger
        # 卷集核的个数
        num_filters = classifier_config['num_filters']
        checkpoints_dir = classifier_config['checkpoints_dir']
        logger.info('loading model parameter')
        if classifier == 'textcnn':
            from engines.models.textcnn import TextCNN
            self.model = TextCNN(seq_length, num_filters, num_classes, embedding_dim)
        elif classifier == 'textrcnn':
            from engines.models.textrcnn import TextRCNN
            self.model = TextRCNN(seq_length, num_classes, hidden_dim, embedding_dim)
        else:
            raise Exception('config model is not exist')
        # 实例化Checkpoint，设置恢复对象为新建立的模型
        checkpoint = tf.train.Checkpoint(model=self.model)
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
        logits = self.model.call(inputs=vector)
        prediction = tf.argmax(logits, axis=-1)
        prediction = prediction.numpy()[0]
        return reverse_classes[prediction]
