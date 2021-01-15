# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py
# @Software: PyCharm
import numpy as np
import time
import math
import tensorflow as tf
from tqdm import tqdm
from engines.utils.focal_loss import FocalLoss
from engines.utils.metrics import cal_metrics
from config import classifier_config

tf.keras.backend.set_floatx('float32')


def train(data_manager, logger):
    embedding_dim = data_manager.embedding_dim
    num_classes = data_manager.max_label_number
    seq_length = data_manager.max_sequence_length

    checkpoints_dir = classifier_config['checkpoints_dir']
    checkpoint_name = classifier_config['checkpoint_name']
    num_filters = classifier_config['num_filters']
    learning_rate = classifier_config['learning_rate']
    epoch = classifier_config['epoch']
    max_to_keep = classifier_config['max_to_keep']
    print_per_batch = classifier_config['print_per_batch']
    is_early_stop = classifier_config['is_early_stop']
    patient = classifier_config['patient']
    hidden_dim = classifier_config['hidden_dim']
    classifier = classifier_config['classifier']

    reverse_classes = {str(class_id): class_name for class_name, class_id in data_manager.class_id.items()}

    best_f1_val = 0.0
    best_at_epoch = 0
    unprocessed = 0
    batch_size = data_manager.batch_size
    very_start_time = time.time()
    loss_obj = FocalLoss() if classifier_config['use_focal_loss'] else None
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    X_train, y_train, X_val, y_val = data_manager.get_training_set()
    # 载入模型
    if classifier == 'textcnn':
        from engines.models.textcnn import TextCNN
        model = TextCNN(seq_length, num_filters, num_classes, embedding_dim)
    elif classifier == 'textrcnn':
        from engines.models.textrcnn import TextRCNN
        model = TextRCNN(seq_length, num_classes, hidden_dim, embedding_dim)
    else:
        raise Exception('config model is not exist')
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoints_dir, checkpoint_name=checkpoint_name, max_to_keep=max_to_keep)
    num_iterations = int(math.ceil(1.0 * len(X_train) / batch_size))
    num_val_iterations = int(math.ceil(1.0 * len(X_val) / batch_size))
    logger.info(('+' * 20) + 'training starting' + ('+' * 20))
    for i in range(epoch):
        start_time = time.time()
        # shuffle train at each epoch
        sh_index = np.arange(len(X_train))
        np.random.shuffle(sh_index)
        X_train = X_train[sh_index]
        y_train = y_train[sh_index]
        logger.info('epoch:{}/{}'.format(i + 1, epoch))
        for iteration in tqdm(range(num_iterations)):
            X_train_batch, y_train_batch = data_manager.next_batch(X_train, y_train, start_index=iteration * batch_size)
            with tf.GradientTape() as tape:
                logits = model.call(X_train_batch, training=1)
                if classifier_config['use_focal_loss']:
                    loss_vec = loss_obj.call(y_true=y_train_batch, y_pred=logits)
                else:
                    loss_vec = tf.keras.losses.categorical_crossentropy(y_true=y_train_batch, y_pred=logits)
                loss = tf.reduce_mean(loss_vec)
            # 定义好参加梯度的参数
            gradients = tape.gradient(loss, model.trainable_variables)
            # 反向传播，自动微分计算
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            if iteration % print_per_batch == 0 and iteration != 0:
                predictions = tf.argmax(logits, axis=-1).numpy()
                y_train_batch = tf.argmax(y_train_batch, axis=-1).numpy()
                measures, _ = cal_metrics(y_true=y_train_batch, y_pred=predictions)
                res_str = ''
                for k, v in measures.items():
                    res_str += (k + ': %.3f ' % v)
                logger.info('training batch: %5d, loss: %.5f, %s' % (iteration, loss, res_str))

        # validation
        logger.info('start evaluate engines...')
        y_true, y_pred = np.array([]), np.array([])

        for iteration in tqdm(range(num_val_iterations)):
            X_val_batch, y_val_batch = data_manager.next_batch(X_val, y_val, iteration * batch_size)
            logits = model.call(X_val_batch)
            predictions = tf.argmax(logits, axis=-1)
            y_val_batch = tf.argmax(y_val_batch, axis=-1)
            y_true = np.append(y_true, y_val_batch)
            y_pred = np.append(y_pred, predictions)

        measures, each_classes = cal_metrics(y_true=y_true, y_pred=y_pred)

        # 打印每一个类别的指标
        classes_val_str = ''
        for k, v in each_classes.items():
            if k in reverse_classes:
                classes_val_str += (reverse_classes[k] + ': ' + str(each_classes[k]) + '\n')
        logger.info(classes_val_str)

        val_res_str = ''
        for k, v in measures.items():
            val_res_str += (k + ': %.3f ' % measures[k])

        time_span = (time.time() - start_time) / 60

        logger.info('time consumption:%.2f(min), %s' % (time_span, val_res_str))
        if measures['f1'] > best_f1_val:
            unprocessed = 0
            best_f1_val = measures['f1']
            best_at_epoch = i + 1
            checkpoint_manager.save()
            logger.info('saved the new best model with f1: %.3f' % best_f1_val)
        else:
            unprocessed += 1

        if is_early_stop:
            if unprocessed >= patient:
                logger.info('early stopped, no progress obtained within {} epochs'.format(patient))
                logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
                logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
                return
    logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
    logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
