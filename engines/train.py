# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : train.py
# @Software: PyCharm
import numpy as np
import time
import pandas as pd
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

    train_file = classifier_config['train_file']
    dev_file = classifier_config['dev_file']
    train_df = pd.read_csv(train_file).sample(frac=1)

    if dev_file is '':
        # split the data into train and validation set
        train_df, dev_df = train_df[:int(len(train_df)*0.9)], train_df[int(len(train_df)*0.9):]
    else:
        dev_df = pd.read_csv(dev_file).sample(frac=1)

    train_dataset = data_manager.get_dataset(train_df, step='train')
    dev_dataset = data_manager.get_dataset(dev_df)

    vocab_size = data_manager.vocab_size

    embedding_method = classifier_config['embedding_method']
    if embedding_method == 'Bert':
        from transformers import TFBertModel
        bert_model = TFBertModel.from_pretrained('bert-base-multilingual-cased')
    else:
        bert_model = None
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

    # 载入模型
    if classifier == 'textcnn':
        from engines.models.textcnn import TextCNN
        model = TextCNN(seq_length, num_filters, num_classes, embedding_dim, vocab_size)
    elif classifier == 'textrcnn':
        from engines.models.textrcnn import TextRCNN
        model = TextRCNN(seq_length, num_classes, hidden_dim, embedding_dim, vocab_size)
    elif classifier == 'textrnn':
        from engines.models.textrnn import TextRNN
        model = TextRNN(seq_length, num_classes, hidden_dim, embedding_dim, vocab_size)
    else:
        raise Exception('config model is not exist')
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoints_dir, checkpoint_name=checkpoint_name, max_to_keep=max_to_keep)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    logger.info(('+' * 20) + 'training starting' + ('+' * 20))
    for i in range(epoch):
        start_time = time.time()
        logger.info('epoch:{}/{}'.format(i + 1, epoch))
        for step, batch in tqdm(train_dataset.shuffle(len(train_dataset)).batch(batch_size).enumerate()):
            if embedding_method == 'Bert':
                X_train_batch, y_train_batch = batch
                X_train_batch = bert_model(X_train_batch)[0]
            else:
                X_train_batch, y_train_batch = batch

            with tf.GradientTape() as tape:
                logits = model(X_train_batch, training=1)
                if classifier_config['use_focal_loss']:
                    loss_vec = loss_obj.call(y_true=y_train_batch, y_pred=logits)
                else:
                    loss_vec = tf.keras.losses.categorical_crossentropy(y_true=y_train_batch, y_pred=logits)
                loss = tf.reduce_mean(loss_vec)
            # 定义好参加梯度的参数
            gradients = tape.gradient(loss, model.trainable_variables)
            # 反向传播，自动微分计算
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            if step % print_per_batch == 0 and step != 0:
                predictions = tf.argmax(logits, axis=-1).numpy()
                y_train_batch = tf.argmax(y_train_batch, axis=-1).numpy()
                measures, _ = cal_metrics(y_true=y_train_batch, y_pred=predictions)
                res_str = ''
                for k, v in measures.items():
                    res_str += (k + ': %.3f ' % v)
                logger.info('training batch: %5d, loss: %.5f, %s' % (step, loss, res_str))

        # validation
        logger.info('start evaluate engines...')
        y_true, y_pred = np.array([]), np.array([])
        loss_values = []

        for dev_batch in tqdm(dev_dataset.batch(batch_size)):
            if embedding_method == 'Bert':
                X_val_batch, y_val_batch = dev_batch
                X_val_batch = bert_model(X_val_batch)[0]
            else:
                X_val_batch, y_val_batch = dev_batch

            logits = model(X_val_batch)
            val_loss_vec = tf.keras.losses.categorical_crossentropy(y_true=y_val_batch, y_pred=logits)
            val_loss = tf.reduce_mean(val_loss_vec)
            predictions = tf.argmax(logits, axis=-1)
            y_val_batch = tf.argmax(y_val_batch, axis=-1)
            y_true = np.append(y_true, y_val_batch)
            y_pred = np.append(y_pred, predictions)
            loss_values.append(val_loss)

        measures, each_classes = cal_metrics(y_true=y_true, y_pred=y_pred)

        # 打印每一个类别的指标
        classes_val_str = ''
        for k, v in each_classes.items():
            if k in reverse_classes:
                classes_val_str += ('\n' + reverse_classes[k] + ': ' + str(each_classes[k]))
        logger.info(classes_val_str)
        # 打印损失函数
        val_res_str = 'loss: %.3f ' % np.mean(loss_values)
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
