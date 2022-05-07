# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : main.py
# @Software: PyCharm
from engines.data import DataManager
from engines.utils.logger import get_logger
from engines.train import Train
from engines.predict import Predictor
from engines.utils.word2vec import Word2VecUtils
from engines.utils.sentence2vec import Sentence2VecUtils
from config import mode, classifier_config, word2vec_config, CUDA_VISIBLE_DEVICES
import tensorflow as tf
import json
import os


def fold_check(config):

    if config['checkpoints_dir'] == '':
        raise Exception('checkpoints_dir did not set...')

    if not os.path.exists(config['checkpoints_dir']):
        print('checkpoints fold not found, creating...')
        os.makedirs(config['checkpoints_dir'])

    if not os.path.exists(config['checkpoints_dir'] + '/logs'):
        print('log fold not found, creating...')
        os.mkdir(config['checkpoints_dir'] + '/logs')


if __name__ == '__main__':
    fold_check(config=classifier_config)
    logger = get_logger(classifier_config['checkpoints_dir'] + '/logs', mode)
    if CUDA_VISIBLE_DEVICES != -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA_VISIBLE_DEVICES)
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[CUDA_VISIBLE_DEVICES], True)
    else:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        
    # 训练分类器
    if mode == 'train_classifier':
        logger.info(json.dumps(classifier_config, indent=2, ensure_ascii=False))
        data_manage = DataManager(logger)
        logger.info('mode: train_classifier')
        logger.info('model: {}'.format(classifier_config['classifier']))
        train = Train(data_manage, logger)
        train.train()
    # 测试分类
    elif mode == 'interactive_predict':
        logger.info(json.dumps(classifier_config, indent=2, ensure_ascii=False))
        data_manage = DataManager(logger)
        logger.info('mode: predict_one')
        logger.info('model: {}'.format(classifier_config['classifier']))
        predictor = Predictor(data_manage, logger)
        predictor.predict_one('warm start')
        while True:
            logger.info('please input a sentence (enter [exit] to exit.)')
            sentence = input()
            if sentence == 'exit':
                break
            results = predictor.predict_one(sentence)
            print(results)
    # 训练词向量
    elif mode == 'train_word2vec':
        logger.info(json.dumps(word2vec_config, indent=2, ensure_ascii=False))
        logger.info('mode: train_word2vec')
        w2v = Word2VecUtils(logger)
        w2v.train_word2vec()
    # 训练SIF句向量
    elif mode == 'train_sif_sentence_vec':
        logger.info(json.dumps(word2vec_config, indent=2, ensure_ascii=False))
        logger.info('mode: train_sif_sentence_vec')
        w2v = Word2VecUtils(logger)
        sif = Sentence2VecUtils(logger)
        sif.train_pca()
    # 批量测试
    elif mode == 'test':
        logger.info('mode: test')
        data_manage = DataManager(logger)
        predictor = Predictor(data_manage, logger)
        predictor.predict_test()
    # 保存pb格式的模型用于tf-severing接口
    elif mode == 'save_model':
        logger.info('mode: save_pb_model')
        data_manage = DataManager(logger)
        predictor = Predictor(data_manage, logger)
        predictor.save_model()
