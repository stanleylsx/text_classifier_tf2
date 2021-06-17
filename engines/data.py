# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
from engines.utils.word2vec import Word2VecUtils
from engines.utils.clean_data import filter_word, filter_char
from config import classifier_config
from collections import Counter


class DataManager:

    def __init__(self, logger):
        self.logger = logger
        self.token_level = classifier_config['token_level']
        self.embedding_method = classifier_config['embedding_method']
        if self.token_level == 'char' and self.embedding_method is not None:
            raise Exception('字粒度不应该使用词嵌入')
        self.w2v_util = Word2VecUtils(logger)
        self.stop_words = self.w2v_util.get_stop_words()

        if self.embedding_method == 'word2vec':
            self.w2v_model = Word2Vec.load(self.w2v_util.model_path)
            self.embedding_dim = self.w2v_model.vector_size
            self.vocab_size = len(self.w2v_model.wv.vocab)
        elif self.embedding_method == 'Bert':
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.embedding_dim = 768
            self.vocab_size = len(self.tokenizer.get_vocab())
        else:
            self.embedding_dim = classifier_config['embedding_dim']
            self.token_file = classifier_config['token_file']
            if not os.path.isfile(self.token_file):
                self.logger.info('vocab files not exist...')
            else:
                self.token2id, self.id2token = self.load_vocab()
                self.vocab_size = len(self.token2id)

        self.PADDING = '[PAD]'
        self.UNKNOWN = '[UNK]'
        self.batch_size = classifier_config['batch_size']
        self.max_sequence_length = classifier_config['max_sequence_length']

        self.class_id = classifier_config['classes']
        self.class_list = [name for name, index in classifier_config['classes'].items()]
        self.max_label_number = len(self.class_id)

        self.logger.info('dataManager initialed...')

    def load_vocab(self, sentences=None):
        if not os.path.isfile(self.token_file):
            self.logger.info('vocab files not exist, building vocab...')
            return self.build_vocab(self.token_file, sentences)
        word_token2id, id2word_token = {}, {}
        with open(self.token_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                word_token, word_token_id = row.split('\t')[0], int(row.split('\t')[1])
                word_token2id[word_token] = word_token_id
                id2word_token[word_token_id] = word_token
        self.vocab_size = len(word_token2id)
        return word_token2id, id2word_token

    def build_vocab(self, token_file, sentences):
        tokens = []
        if self.token_level == 'word':
            # 词粒度
            for sentence in tqdm(sentences):
                words = self.w2v_util.processing_sentence(sentence, self.stop_words)
                tokens.extend(words)
            # 根据词频过滤一部分频率极低的词/字，不加入词表
            count_dict = Counter(tokens)
            tokens = [k for k, v in count_dict.items() if v > 1 and filter_word(k)]
        else:
            # 字粒度
            for sentence in tqdm(sentences):
                chars = list(sentence)
                tokens.extend(chars)
            # 根据词频过滤一部分频率极低的词/字，不加入词表
            count_dict = Counter(tokens)
            tokens = [k for k, v in count_dict.items() if k != ' ' and filter_char(k)]
        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        # 向生成的词表和标签表中加入[PAD]
        id2token[0] = self.PADDING
        token2id[self.PADDING] = 0
        # 向生成的词表中加入[UNK]
        id2token[len(id2token)] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(id2token)
        # 保存词表及标签表
        with open(token_file, 'w', encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + '\t' + str(idx) + '\n')
        self.vocab_size = len(token2id)
        return token2id, id2token

    def padding(self, sentence):
        """
        长度不足max_sequence_length则补齐
        :param sentence:
        :return:
        """
        if len(sentence) < self.max_sequence_length:
            sentence += [self.PADDING for _ in range(self.max_sequence_length - len(sentence))]
        else:
            sentence = sentence[:self.max_sequence_length]
        return sentence

    def prepare_w2v_data(self, sentences, labels):
        """
        输出word2vec做embedding之后的X矩阵和y向量
        """
        self.logger.info('loading data...')
        X, y = [], []
        embedding_unknown = [0] * self.embedding_dim
        for record in tqdm(zip(sentences, labels)):
            sentence = self.w2v_util.processing_sentence(record[0], self.stop_words)
            sentence = self.padding(sentence)
            label = tf.one_hot(record[1], depth=self.max_label_number)
            vector = []
            for word in sentence:
                if word in self.w2v_model.wv.vocab:
                    vector.append(self.w2v_model[word].tolist())
                else:
                    vector.append(embedding_unknown)
            X.append(vector)
            y.append(label)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def prepare_bert_data(self, sentences, labels):
        """
        输出Bert做embedding之后的X矩阵和y向量
        """
        self.logger.info('loading data...')
        tokens_list, y = [], []
        for record in tqdm(zip(sentences, labels)):
            label = tf.one_hot(record[1], depth=self.max_label_number)
            if len(record[0]) > self.max_sequence_length-2:
                sentence = record[0][:self.max_sequence_length-2]
                tokens = self.tokenizer.encode(sentence)
            else:
                tokens = self.tokenizer.encode(record[0])
            if len(tokens) < self.max_sequence_length:
                tokens += [0 for _ in range(self.max_sequence_length - len(tokens))]
            tokens_list.append(tokens)
            y.append(label)
        return np.array(tokens_list), np.array(y, dtype=np.float32)

    def prepare_data(self, sentences, labels):
        """
        输出X矩阵和y向量
        """
        self.logger.info('loading data...')
        X, y = [], []
        for record in tqdm(zip(sentences, labels)):
            if self.token_level == 'word':
                sentence = self.w2v_util.processing_sentence(record[0], self.stop_words)
                sentence = self.padding(sentence)
            else:
                sentence = list(record[0])
                sentence = self.padding(sentence)
            label = tf.one_hot(record[1], depth=self.max_label_number)
            tokens = []
            for word in sentence:
                if word in self.token2id:
                    tokens.append(self.token2id[word])
                else:
                    tokens.append(self.token2id[self.UNKNOWN])
            X.append(tokens)
            y.append(label)
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def get_dataset(self, df, step=None):
        """
        构建Dataset
        """
        df = df.loc[df.label.isin(self.class_list)]
        df['label'] = df.label.map(lambda x: self.class_id[x])
        # convert the data in matrix
        if self.token_level == 'word':
            if self.embedding_method == 'word2vec':
                X, y = self.prepare_w2v_data(df['sentence'], df['label'])
            elif self.embedding_method == 'Bert':
                X, y = self.prepare_bert_data(df['sentence'], df['label'])
            else:
                if step == 'train' and not os.path.isfile(self.token_file):
                    self.token2id, self.id2token = self.load_vocab(df['sentence'])
                X, y = self.prepare_data(df['sentence'], df['label'])
        else:
            if step == 'train' and not os.path.isfile(self.token_file):
                self.token2id, self.id2token = self.load_vocab(df['sentence'])
            X, y = self.prepare_data(df['sentence'], df['label'])
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset

    def prepare_single_sentence(self, sentence):
        """
        把预测的句子转成矩阵和向量
        :param sentence:
        :return:
        """
        if self.token_level == 'word':
            if self.embedding_method == 'word2vec':
                embedding_unknown = [0] * self.embedding_dim
                sentence = self.w2v_util.processing_sentence(sentence, self.stop_words)
                sentence = self.padding(sentence)
                vector = []
                for word in sentence:
                    if word in self.w2v_model.wv.vocab:
                        vector.append(self.w2v_model[word].tolist())
                    else:
                        vector.append(embedding_unknown)
                return np.array([vector], dtype=np.float32)
            elif self.embedding_method == 'Bert':
                if len(sentence) > self.max_sequence_length - 2:
                    sentence = sentence[:self.max_sequence_length - 2]
                    tokens = self.tokenizer.encode(sentence)
                else:
                    tokens = self.tokenizer.encode(sentence)
                if len(tokens) < 150:
                    tokens += [0 for _ in range(self.max_sequence_length - len(tokens))]
                return np.array([tokens])
            else:
                sentence = self.w2v_util.processing_sentence(sentence, self.stop_words)
                sentence = self.padding(sentence)
                word_tokens = []
                for word in sentence:
                    if word in self.token2id:
                        word_tokens.append(self.token2id[word])
                    else:
                        word_tokens.append(self.token2id[self.UNKNOWN])
                return np.array([word_tokens], dtype=np.float32)
        else:
            sentence = list(sentence)
            sentence = self.padding(sentence)
            char_tokens = []
            for word in sentence:
                if word in self.token2id:
                    char_tokens.append(self.token2id[word])
                else:
                    char_tokens.append(self.token2id[self.UNKNOWN])
            return np.array([char_tokens], dtype=np.float32)

