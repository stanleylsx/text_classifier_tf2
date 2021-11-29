# -*- coding: utf-8 -*-
# @Time : 2020/11/25 7:51 下午 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : focal_loss.py 
# @Software: PyCharm
import tensorflow as tf


class FocalLoss(tf.keras.losses.Loss):
    """
    focal loss for multi-class classification
    fl(pt) = -alpha*(1-pt)^(gamma)*log(pt)
    :param alpha:
    :param gamma:
    :param epsilon:
    """
    def __init__(self, gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        # To avoid divided by zero
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        """
        :param y_true: ground truth one-hot vector shape of [batch_size, nb_class]
        :param y_pred: prediction after softmax shape of [batch_size, nb_class]
        :return:
        """
        y_pred = tf.add(y_pred, self.epsilon)
        # Cross entropy
        ce = -y_true * tf.math.log(y_pred)
        # Not necessary to multiply y_true(cause it will multiply with CE which has set unconcerned index to zero ),
        # but refer to the definition of p_t, we do it
        weight = tf.math.pow(1 - y_pred, self.gamma) * y_true
        # Now fl has a shape of [batch_size, nb_class]
        # alpha should be a step function as paper mentioned, but it doesn't matter like reason mentioned above
        # (CE has set unconcerned index to zero)
        # alpha_step = tf.where(y_true, alpha*np.ones_like(y_true), 1-alpha*np.ones_like(y_true))
        fl = ce * weight * self.alpha
        loss = tf.reduce_sum(fl, axis=1)
        return loss
