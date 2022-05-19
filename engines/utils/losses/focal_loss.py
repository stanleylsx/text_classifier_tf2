# -*- coding: utf-8 -*-
# @Time : 2020/11/25 7:51 下午 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : focal_loss.py 
# @Software: PyCharm
import tensorflow as tf
from config import classifier_config


class FocalLoss(tf.keras.losses.Loss):
    """
    Multi-class Focal loss implementation
    """
    def __init__(self, gamma=2.0, epsilon=1.e-9):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        weight = classifier_config['weight']
        self.alpha = tf.reshape(weight, [-1]) if weight else None
        self.epsilon = epsilon
        self.num_labels = len(classifier_config['classes'])

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        labels = tf.expand_dims(tf.argmax(y_true, axis=-1), axis=-1)
        pt = tf.gather(y_pred, labels, batch_dims=1)
        pt = tf.clip_by_value(pt, self.epsilon, 1. - self.epsilon)
        loss = -tf.multiply(tf.pow(tf.subtract(1., pt), self.gamma), tf.math.log(pt))
        if self.alpha is not None:
            alpha = tf.gather(self.alpha, labels, batch_dims=0)
            loss = tf.multiply(alpha, loss)
        return loss
