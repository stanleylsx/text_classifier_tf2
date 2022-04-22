# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : predict.py
# @Software: PyCharm
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import datetime
from tqdm import tqdm
from engines.utils.metrics import cal_metrics
from config import classifier_config
from texttable import Texttable
from collections import Counter


class Predictor:
    def __init__(self, data_manager, logger):
        hidden_dim = classifier_config['hidden_dim']
        classifier = classifier_config['classifier']
        self.dataManager = data_manager
        self.seq_length = data_manager.max_sequence_length
        num_classes = data_manager.max_label_number
        self.embedding_dim = data_manager.embedding_dim
        vocab_size = data_manager.vocab_size
        self.logger = logger
        self.reverse_classes = data_manager.reverse_classes
        # 卷集核的个数
        num_filters = classifier_config['num_filters']
        self.checkpoints_dir = classifier_config['checkpoints_dir']
        logger.info('loading model parameter')

        if classifier_config['embedding_method'] == 'word2vec':
            embeddings_matrix = data_manager.embeddings_matrix
        else:
            embeddings_matrix = None

        if classifier == 'textcnn':
            from engines.models.TextCNN import TextCNN
            self.model = TextCNN(self.seq_length, num_filters, num_classes, self.embedding_dim, vocab_size,
                                 embeddings_matrix)
        elif classifier == 'textrcnn':
            from engines.models.TextRCNN import TextRCNN
            self.model = TextRCNN(num_classes, hidden_dim, self.embedding_dim, vocab_size, embeddings_matrix)
        elif classifier == 'textrnn':
            from engines.models.TextRNN import TextRNN
            self.model = TextRNN(num_classes, hidden_dim, self.embedding_dim, vocab_size, embeddings_matrix)
        elif classifier == 'transformer':
            from engines.models.Transformer import Transformer
            self.model = Transformer(self.seq_length, num_classes, self.embedding_dim, vocab_size, embeddings_matrix)
        elif classifier == 'Bert':
            from engines.models.Bert import BertClassification
            self.model = BertClassification(num_classes)
        else:
            self.model = None
            raise Exception('config model is not exist')
        # 实例化Checkpoint，设置恢复对象为新建立的模型
        checkpoint = tf.train.Checkpoint(model=self.model)
        # 从文件恢复模型参数
        checkpoint.restore(tf.train.latest_checkpoint(self.checkpoints_dir))
        logger.info('loading model successfully')

    def predict_test(self):
        test_file = classifier_config['test_file']
        if test_file == '':
            self.logger.info('test dataset does not exist!')
            return
        test_df = pd.read_csv(test_file)
        test_dataset = self.dataManager.get_dataset(test_df)
        batch_size = self.dataManager.batch_size
        y_true, y_pred, probabilities = np.array([]), np.array([]), np.array([])
        start_time = time.time()
        for step, batch in tqdm(test_dataset.batch(batch_size).enumerate()):
            X_test_batch, y_test_batch = batch
            logits = self.model(X_test_batch)
            predictions = tf.argmax(logits, axis=-1)
            y_test_batch = tf.argmax(y_test_batch, axis=-1)
            y_true = np.append(y_true, y_test_batch)
            y_pred = np.append(y_pred, predictions)
            max_probability = tf.reduce_max(logits, axis=-1)
            probabilities = np.append(probabilities, max_probability)
        self.logger.info('test time consumption: %.3f(ms)' % ((time.time() - start_time) * 1000))
        measures, each_classes = cal_metrics(y_true=y_true, y_pred=y_pred)
        count_map = {self.reverse_classes[str(int(key))]: value for key, value in Counter(y_true).items()}
        # 打印不一致的下标
        inconsistent = np.argwhere(y_true != y_pred)
        if len(inconsistent) > 0:
            indices = [i for i in list(inconsistent.ravel())]
            y_error_pred = [self.reverse_classes[str(int(i))] for i in list(y_pred[indices])]
            data_dict = {'indices': indices, 'y_error_pred': y_error_pred}
            for col_name in test_df.columns.values.tolist():
                data_dict[col_name] = test_df.iloc[indices][col_name].tolist()
            data_dict['probability'] = list(probabilities[indices])
            indices = [i + 1 for i in indices]
            test_result_file = './logs/' + (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.csv'))
            result = pd.DataFrame(data_dict)
            statics = pd.DataFrame(
                result.groupby('label').apply(lambda data: data.y_error_pred.value_counts())).reset_index()
            statics = statics.rename(columns={'level_1': 'error_predict', 'y_error_pred': 'count'})
            statics['error_rate'] = statics.apply(lambda row: row['count'] / count_map[row['label']], axis=1)
            tb = Texttable()
            tb.set_cols_align(['l', 'r', 'r', 'r'])
            tb.set_cols_dtype(['t', 't', 'i', 'f'])
            tb.header(list(statics.columns))
            tb.add_rows(statics.values, header=False)
            result.to_csv(test_result_file, encoding='utf-8', index=False)
            self.logger.info('\nerror indices in test dataset:')
            self.logger.info(indices)
            self.logger.info('\nerror distribution:')
            self.logger.info(tb.draw())
        # 打印总的指标
        res_str = ''
        for k, v in measures.items():
            res_str += (k + ': %.3f ' % v)
        self.logger.info('%s' % res_str)
        # 打印每一个类别的指标
        classes_val_str = ''
        for k, v in each_classes.items():
            if k in self.reverse_classes:
                classes_val_str += ('\n' + self.reverse_classes[k] + ': ' + str(each_classes[k]))
        self.logger.info(classes_val_str)

    def predict_one(self, sentence):
        """
        对输入的句子分类预测
        :param sentence:
        :return:
        """
        start_time = time.time()
        vector = self.dataManager.prepare_single_sentence(sentence)
        logits = self.model(inputs=vector)
        prediction = tf.argmax(logits, axis=-1)
        prediction = prediction.numpy()[0]
        self.logger.info('predict time consumption: %.3f(ms)' % ((time.time() - start_time)*1000))
        return self.reverse_classes[str(prediction)], logits

    def save_model(self):
        tf.saved_model.save(self.model, self.checkpoints_dir,
                            signatures=self.model.call.get_concrete_function(
                                tf.TensorSpec([None, self.seq_length], tf.int32, name='inputs')))
        self.logger.info('The model has been saved')
