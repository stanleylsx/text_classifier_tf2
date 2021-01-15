# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : metrics.py 
# @Software: PyCharm
from sklearn import metrics
from config import classifier_config


def cal_metrics(y_true, y_pred):
    """
    指标计算
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    average = classifier_config['metrics_average']
    precision = metrics.precision_score(y_true, y_pred, average=average)
    recall = metrics.recall_score(y_true, y_pred, average=average)
    f1 = metrics.f1_score(y_true, y_pred, average=average)
    each_classes = metrics.classification_report(y_true, y_pred, output_dict=True)
    return {'precision': precision, 'recall': recall, 'f1': f1}, each_classes
