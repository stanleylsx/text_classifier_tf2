# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : main.py
# @Software: PyCharm
from engines.data import DataManager
from engines.utils.logger import get_logger
from engines.train import train
from engines.predict import Predictor
from engines.word2vec_util import Word2VecUtils
from config import mode


if __name__ == '__main__':
    logger = get_logger('./logs')
    # 训练分类器
    if mode == 'train_classifier':
        data_manage = DataManager(logger)
        logger.info('mode: train_classifier')
        train(data_manage, logger)
    # 测试分类
    elif mode == 'interactive_predict':
        data_manage = DataManager(logger)
        logger.info('mode: predict_one')
        predictor = Predictor(data_manage, logger)
        while True:
            logger.info('please input a sentence (enter [exit] to exit.)')
            sentence = input()
            if sentence == 'exit':
                break
            results = predictor.predict_one(sentence)
            print(results)
    # 训练词向量
    elif mode == 'train_word2vec':
        logger.info('mode: train_word2vec')
        w2v = Word2VecUtils(logger)
        w2v.train_word2vec()
