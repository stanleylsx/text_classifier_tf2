# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : metrics.py 
# @Software: PyCharm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from config import classifier_config
import numpy as np


def cal_metrics(y_true, y_pred):
    """
    指标计算
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    average = classifier_config['metrics_average']
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    each_classes = classification_report(y_true, y_pred, output_dict=True, labels=np.unique(y_pred),
                                                 zero_division=0)
    return {'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy}, each_classes
