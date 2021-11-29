# -*- coding: utf-8 -*-
# @Time : 2021/11/26 7:51 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : r_drop_loss.py
# @Software: PyCharm
import tensorflow as tf


class RDropLoss:
    """
    r drop loss
    """
    def __init__(self):
        super(RDropLoss, self).__init__()
        self.alpha = 4
        self.kl_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE, name='kl_divergence')
        self.ce_loss = tf.keras.losses.CategoricalCrossentropy()

    def calculate_loss(self, p, q, y_true):
        ce_loss_1 = self.ce_loss(y_true=y_true, y_pred=p)
        ce_loss_2 = self.ce_loss(y_true=y_true, y_pred=q)
        ce_loss = 0.5 * (ce_loss_1 + ce_loss_2)
        ce_loss = tf.reduce_mean(ce_loss)

        kl_loss_1 = self.kl_loss(p, q)
        kl_loss_2 = self.kl_loss(q, p)
        kl_loss = 0.5 * (kl_loss_1 + kl_loss_2)
        kl_loss = tf.reduce_mean(kl_loss)

        loss = ce_loss + self.alpha * kl_loss
        return loss






