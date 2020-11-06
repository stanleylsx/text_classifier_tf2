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
    model = classifier_config['model']

    best_f1_val = 0.0
    best_at_epoch = 0
    unprocessed = 0
    batch_size = data_manager.batch_size
    very_start_time = time.time()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    X_train, y_train, X_val, y_val = data_manager.get_training_set()
    # 载入模型
    if model == 'textcnn':
        from engines.models.textcnn import TextCNN
        model = TextCNN(seq_length, num_filters, num_classes, embedding_dim)
    elif model == 'textrcnn':
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
                loss_vec = tf.keras.losses.sparse_categorical_crossentropy(y_pred=logits, y_true=y_train_batch)
                loss = tf.reduce_mean(loss_vec)
            # 定义好参加梯度的参数
            gradients = tape.gradient(loss, model.trainable_variables)
            # 反向传播，自动微分计算
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            if iteration % print_per_batch == 0 and iteration != 0:
                predictions = tf.argmax(logits, axis=-1)
                measures = cal_metrics(y_true=y_train_batch, y_pred=predictions)
                res_str = ''
                for k, v in measures.items():
                    res_str += (k + ': %.3f ' % v)
                logger.info('training batch: %5d, loss: %.5f, %s' % (iteration, loss, res_str))

        # validation
        logger.info('start evaluate engines...')
        val_results = {'precision': 0, 'recall': 0, 'f1': 0}
        for iteration in tqdm(range(num_val_iterations)):
            X_val_batch, y_val_batch = data_manager.next_batch(X_val, y_val, iteration * batch_size)
            logits = model.call(X_val_batch)
            predictions = tf.argmax(logits, axis=-1)
            measures = cal_metrics(y_true=y_val_batch, y_pred=predictions)
            for k, v in measures.items():
                val_results[k] += v

        time_span = (time.time() - start_time) / 60
        val_res_str = ''
        dev_f1_avg = 0
        for k, v in val_results.items():
            val_results[k] /= num_val_iterations
            val_res_str += (k + ': %.3f ' % val_results[k])
            if k == 'f1':
                dev_f1_avg = val_results[k]
        logger.info('time consumption:%.2f(min), %s' % (time_span, val_res_str))

        if np.array(dev_f1_avg).mean() > best_f1_val:
            unprocessed = 0
            best_f1_val = np.array(dev_f1_avg).mean()
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
