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
from tensorflow.keras.losses import CategoricalCrossentropy
tf.keras.backend.set_floatx('float32')


class Train:
    def __init__(self, data_manager, logger):
        self.logger = logger
        self.data_manager = data_manager
        self.reverse_classes = data_manager.reverse_classes

        self.epoch = classifier_config['epoch']
        self.print_per_batch = classifier_config['print_per_batch']
        self.is_early_stop = classifier_config['is_early_stop']
        self.patient = classifier_config['patient']
        self.use_gan = classifier_config['use_gan']
        self.gan_method = classifier_config['gan_method']
        self.batch_size = self.data_manager.batch_size
        self.loss_function = FocalLoss() if classifier_config['use_focal_loss'] else CategoricalCrossentropy()
        if classifier_config['use_r_drop']:
            from engines.utils.r_drop_loss import RDropLoss
            self.r_drop_loss = RDropLoss()

        learning_rate = classifier_config['learning_rate']
        if classifier_config['optimizer'] == 'Adagrad':
            self.optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif classifier_config['optimizer'] == 'Adadelta':
            self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        elif classifier_config['optimizer'] == 'RMSprop':
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif classifier_config['optimizer'] == 'SGD':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif classifier_config['optimizer'] == 'Adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif classifier_config['optimizer'] == 'AdamW':
            from tensorflow_addons.optimizers import AdamW
            self.optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-2)
        else:
            raise Exception('optimizer does not exist')

        # 加入word2vec进行训练
        if classifier_config['embedding_method'] == 'word2vec':
            embeddings_matrix = data_manager.embeddings_matrix
        else:
            embeddings_matrix = None
        classifier = classifier_config['classifier']
        embedding_dim = data_manager.embedding_dim
        num_classes = data_manager.max_label_number
        vocab_size = data_manager.vocab_size
        # 载入模型
        if classifier == 'TextCNN':
            from engines.models.TextCNN import TextCNN
            self.model = TextCNN(num_classes, embedding_dim, vocab_size, embeddings_matrix)
        elif classifier == 'TextRCNN':
            from engines.models.TextRCNN import TextRCNN
            self.model = TextRCNN(num_classes, embedding_dim, vocab_size, embeddings_matrix)
        elif classifier == 'TextRNN':
            from engines.models.TextRNN import TextRNN
            self.model = TextRNN(num_classes, embedding_dim, vocab_size, embeddings_matrix)
        elif classifier == 'Transformer':
            from engines.models.Transformer import Transformer
            self.model = Transformer(num_classes, embedding_dim, vocab_size, embeddings_matrix)
        elif classifier == 'Bert':
            from engines.models.Bert import BertClassification
            self.model = BertClassification(num_classes)
        elif classifier == 'DistilBert':
            from engines.models.DistilBert import DistilBertClassification
            self.model = DistilBertClassification(num_classes)
        elif classifier == 'AlBert':
            from engines.models.AlBert import AlBertClassification
            self.model = AlBertClassification(num_classes)
        elif classifier == 'RoBerta':
            from engines.models.RoBerta import RoBertaClassification
            self.model = RoBertaClassification(num_classes)
        elif classifier == 'Electra':
            from engines.models.Electra import ElectraClassification
            self.model = ElectraClassification(num_classes)
        elif classifier == 'XLNet':
            from engines.models.XLNet import XLNetClassification
            self.model = XLNetClassification(num_classes)
        else:
            raise Exception('config model is not exist')

        checkpoints_dir = classifier_config['checkpoints_dir']
        checkpoint_name = classifier_config['checkpoint_name']
        max_to_keep = classifier_config['max_to_keep']
        checkpoint = tf.train.Checkpoint(model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoints_dir, checkpoint_name=checkpoint_name, max_to_keep=max_to_keep)
        checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        if self.checkpoint_manager.latest_checkpoint:
            print('Restored from {}'.format(self.checkpoint_manager.latest_checkpoint))
        else:
            print('Initializing from scratch.')

    def train(self):
        train_file = classifier_config['train_file']
        val_file = classifier_config['val_file']
        train_df = pd.read_csv(train_file).sample(frac=1)

        if val_file == '':
            self.logger.info('generate validation dataset...')
            validation_rate = 0.15
            ratio = 1 - validation_rate
            # split the data into train and validation set
            train_df, val_df = train_df[:int(len(train_df) * ratio)], train_df[int(len(train_df) * ratio):]
            val_df = val_df.sample(frac=1)
        else:
            val_df = pd.read_csv(val_file).sample(frac=1)

        train_dataset = self.data_manager.get_dataset(train_df, step='train')
        val_dataset = self.data_manager.get_dataset(val_df)

        best_f1_val = 0.0
        best_at_epoch = 0
        unprocessed = 0
        very_start_time = time.time()

        self.logger.info(('+' * 20) + 'training starting' + ('+' * 20))
        for i in range(self.epoch):
            start_time = time.time()
            self.logger.info('\nepoch:{}/{}'.format(i + 1, self.epoch))
            for step, batch in tqdm(train_dataset.shuffle(len(train_dataset)).batch(self.batch_size).enumerate()):
                X_train_batch, y_train_batch = batch
                with tf.GradientTape() as tape:
                    logits = self.model(X_train_batch, training=1)
                    if classifier_config['use_r_drop']:
                        logits_2 = self.model(X_train_batch, training=1)
                        loss = self.r_drop_loss.calculate_loss(logits, logits_2, y_train_batch)
                    else:
                        loss_vec = self.loss_function(y_true=y_train_batch, y_pred=logits)
                        loss = tf.reduce_mean(loss_vec)
                # 定义好参加梯度的参数
                variables = self.model.trainable_variables
                # 将预训练模型里面的pooler层的参数去掉
                variables = [var for var in variables if 'pooler' not in var.name]
                gradients = tape.gradient(loss, variables)

                if self.use_gan:
                    if self.gan_method == 'fgm':
                        # 使用FGM的对抗办法
                        epsilon = 1.0
                        embedding = variables[0]
                        embedding_gradients = gradients[0]
                        embedding_gradients = tf.zeros_like(embedding) + embedding_gradients
                        delta = epsilon * embedding_gradients / tf.norm(embedding_gradients, ord=2)

                        accum_vars = [tf.Variable(tf.zeros_like(grad), trainable=False) for grad in gradients]
                        gradients = [accum_vars[i].assign_add(grad) for i, grad in enumerate(gradients)]
                        variables[0].assign_add(delta)

                        with tf.GradientTape() as gan_tape:
                            logits = self.model(X_train_batch, training=1)
                            if classifier_config['use_r_drop']:
                                logits_2 = self.model(X_train_batch, training=1)
                                loss = self.r_drop_loss.calculate_loss(logits, logits_2, y_train_batch)
                            else:
                                loss_vec = self.loss_function(y_true=y_train_batch, y_pred=logits)
                                loss = tf.reduce_mean(loss_vec)
                        gan_gradients = gan_tape.gradient(loss, variables)
                        gradients = [gradients[i].assign_add(grad) for i, grad in enumerate(gan_gradients)]
                        variables[0].assign_sub(delta)
                    elif self.gan_method == 'pgd':
                        # 使用PGD的对抗办法
                        alpha = 0.3
                        epsilon = 1
                        k = classifier_config['attack_round']
                        origin_embedding = tf.Variable(variables[0])
                        accum_vars = [tf.Variable(tf.zeros_like(grad), trainable=False) for grad in gradients]
                        origin_gradients = [accum_vars[i].assign_add(grad) for i, grad in enumerate(gradients)]

                        for t in range(k):
                            embedding = variables[0]
                            embedding_gradients = gradients[0]
                            embedding_gradients = tf.zeros_like(embedding) + embedding_gradients
                            delta = alpha * embedding_gradients / tf.norm(embedding_gradients, ord=2)
                            variables[0].assign_add(delta)

                            r = variables[0] - origin_embedding
                            if tf.norm(r, ord=2) > epsilon:
                                r = epsilon * r / tf.norm(r, ord=2)
                            variables[0].assign(origin_embedding + tf.Variable(r))

                            if t != k - 1:
                                gradients = [tf.Variable(tf.zeros_like(grad), trainable=False) for grad in gradients]
                            else:
                                gradients = origin_gradients
                            with tf.GradientTape() as gan_tape:
                                logits = self.model(X_train_batch, training=1)
                                if classifier_config['use_r_drop']:
                                    logits_2 = self.model(X_train_batch, training=1)
                                    loss = self.r_drop_loss.calculate_loss(logits, logits_2, y_train_batch)
                                else:
                                    loss_vec = self.loss_function(y_true=y_train_batch, y_pred=logits)
                                    loss = tf.reduce_mean(loss_vec)
                            gan_gradients = gan_tape.gradient(loss, variables)
                            gradients = [gradients[i].assign_add(grad) for i, grad in enumerate(gan_gradients)]
                        variables[0].assign(origin_embedding)

                # 反向传播，自动微分计算
                self.optimizer.apply_gradients(zip(gradients, variables))
                if step % self.print_per_batch == 0 and step != 0:
                    predictions = tf.argmax(logits, axis=-1).numpy()
                    y_train_batch = tf.argmax(y_train_batch, axis=-1).numpy()
                    measures, _ = cal_metrics(y_true=y_train_batch, y_pred=predictions)
                    res_str = ''
                    for k, v in measures.items():
                        res_str += (k + ': %.3f ' % v)
                    self.logger.info('training batch: %5d, loss: %.5f, %s' % (step, loss, res_str))

            measures = self.validate(val_dataset)
            time_span = (time.time() - start_time) / 60
            self.logger.info('time consumption:%.2f(min)' % time_span)

            if measures['f1'] > best_f1_val:
                unprocessed = 0
                best_f1_val = measures['f1']
                best_at_epoch = i + 1
                self.checkpoint_manager.save()
                self.logger.info('saved the new best model with f1: %.3f' % best_f1_val)
            else:
                unprocessed += 1

            if self.is_early_stop:
                if unprocessed >= self.patient:
                    self.logger.info('early stopped, no progress obtained within {} epochs'.format(self.patient))
                    self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
                    self.logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))
                    return
        self.logger.info('overall best f1 is {} at {} epoch'.format(best_f1_val, best_at_epoch))
        self.logger.info('total training time consumption: %.3f(min)' % ((time.time() - very_start_time) / 60))

    def validate(self, val_dataset):
        self.logger.info('start evaluate engines...')
        y_true, y_pred = np.array([]), np.array([])
        loss_values = []
        for val_batch in tqdm(val_dataset.batch(self.batch_size)):
            X_val_batch, y_val_batch = val_batch
            logits = self.model(X_val_batch)
            val_loss_vec = self.loss_function(y_true=y_val_batch, y_pred=logits)
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
            if k in self.reverse_classes:
                classes_val_str += ('\n' + self.reverse_classes[k] + ': ' + str(each_classes[k]))
        self.logger.info(classes_val_str)
        # 打印损失函数
        val_res_str = 'loss: %.3f ' % np.mean(loss_values)
        for k, v in measures.items():
            val_res_str += (k + ': %.3f ' % measures[k])
        self.logger.info(val_res_str)
        return measures
