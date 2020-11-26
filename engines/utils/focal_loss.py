# -*- coding: utf-8 -*-
# @Time : 2020/11/25 7:51 下午 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : focal_loss.py 
# @Software: PyCharm
import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, tf.float32)
        loss = -y_true * tf.math.pow(1 - y_pred, self.gamma) * tf.math.log(y_pred)
        loss = tf.reduce_sum(loss, axis=1)
        return loss
