# -*- coding: utf-8 -*-
# @Time : 2021/11/26 7:51 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : r_drop_loss.py
# @Software: PyCharm
import tensorflow as tf
from config import classifier_config
from engines.utils.losses.focal_loss import FocalLoss
from tensorflow.keras.losses import CategoricalCrossentropy


class RDropLoss:
    """
    r drop loss
    """
    def __init__(self):
        super(RDropLoss, self).__init__()
        self.alpha = 4
        self.kl_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE, name='kl_divergence')
        self.loss = FocalLoss() if classifier_config['use_focal_loss'] else CategoricalCrossentropy()

    def calculate_loss(self, p, q, y_true):
        loss_1 = self.loss(y_true=y_true, y_pred=p)
        loss_2 = self.loss(y_true=y_true, y_pred=q)
        loss = 0.5 * (loss_1 + loss_2)
        loss = tf.reduce_mean(loss)

        kl_loss_1 = self.kl_loss(p, q)
        kl_loss_2 = self.kl_loss(q, p)
        kl_loss = 0.5 * (kl_loss_1 + kl_loss_2)
        kl_loss = tf.reduce_mean(kl_loss)

        loss = loss + self.alpha * kl_loss
        return loss
