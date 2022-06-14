# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : data.py
# @Software: PyCharm
import numpy as np
import tensorflow as tf
import os
import jieba
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
        self.classifier = classifier_config['classifier']
        self.token_file = classifier_config['token_file']
        self.support_pretrained_model = ['Bert', 'DistilBert', 'AlBert', 'Electra', 'RoBerta', 'XLNet']
        if self.embedding_method != '':
            if self.classifier in self.support_pretrained_model:
                raise Exception('如果使用预训练模型微调，不需要设定embedding_method')
        if self.token_level == 'char' and self.embedding_method != '':
            raise Exception('字粒度不应该使用词嵌入')
        self.w2v_util = Word2VecUtils(logger)
        if classifier_config['stop_words']:
            self.stop_words = self.w2v_util.get_stop_words()
        else:
            self.stop_words = []
        self.remove_sp = True if classifier_config['remove_special'] else False
        self.PADDING = '[PAD]'
        self.UNKNOWN = '[UNK]'

        if self.classifier in self.support_pretrained_model:
            self.tokenizer = self.tokenizer_for_pretrained_model(self.classifier)
            self.vocab_size = len(self.tokenizer)
        else:
            if self.embedding_method == 'word2vec':
                self.w2v_model = Word2Vec.load(self.w2v_util.model_path)
                self.embedding_dim = self.w2v_model.vector_size
                self.vocab_size = len(self.w2v_model.wv.vocab)
                self.word2token = {self.PADDING: 0}
                # 所有词对应的嵌入向量 [(word, vector)]
                vocab_list = [(k, self.w2v_model.wv[k]) for k, v in self.w2v_model.wv.vocab.items()]
                # [len(vocab)+1, embedding_dim] '+1'是增加了一个'PAD'
                self.embeddings_matrix = np.zeros((len(self.w2v_model.wv.vocab.items()) + 1, self.w2v_model.vector_size))
                for i in tqdm(range(len(vocab_list))):
                    word = vocab_list[i][0]
                    self.word2token[word] = i + 1
                    self.embeddings_matrix[i + 1] = vocab_list[i][1]
                # 保存词表及标签表
                with open(self.token_file, 'w', encoding='utf-8') as outfile:
                    for word, token in self.word2token.items():
                        outfile.write(word + '\t' + str(token) + '\n')
            else:
                self.embedding_dim = classifier_config['embedding_dim']
                self.token_file = classifier_config['token_file']
                if not os.path.isfile(self.token_file):
                    self.logger.info('vocab files not exist...')
                else:
                    self.token2id, self.id2token = self.load_vocab()

        self.batch_size = classifier_config['batch_size']
        self.max_sequence_length = classifier_config['max_sequence_length']

        self.classes = classifier_config['classes']
        self.class_id = {cls: index for index, cls in enumerate(self.classes)}
        self.max_label_number = len(self.classes)
        self.reverse_classes = {str(class_id): class_name for class_name, class_id in self.class_id.items()}

        self.logger.info('dataManager initialed...')

    def load_vocab(self, sentences=None):
        if not os.path.exists(self.token_file):
            self.logger.info('vocab files not exist, building vocab...')
            return self.build_vocab(sentences)
        word_token2id, id2word_token = {}, {}
        with open(self.token_file, 'r', encoding='utf-8') as infile:
            for row in infile:
                row = row.strip()
                word_token, word_token_id = row.split('\t')[0], int(row.split('\t')[1])
                word_token2id[word_token] = word_token_id
                id2word_token[word_token_id] = word_token
        self.vocab_size = len(word_token2id)
        return word_token2id, id2word_token

    @staticmethod
    def processing_sentence(x, stop_words):
        cut_word = jieba.cut(str(x).strip())
        if stop_words:
            words = [word for word in cut_word if word not in stop_words and word != ' ']
        else:
            words = list(cut_word)
            words = [word for word in words if word != ' ']
        return words

    def build_vocab(self, sentences):
        tokens = []
        if self.token_level == 'word':
            # 词粒度
            for sentence in tqdm(sentences):
                words = self.processing_sentence(sentence, self.stop_words)
                tokens.extend(words)
            # 根据词频过滤一部分频率极低的词/字，不加入词表
            count_dict = Counter(tokens)
            tokens = [k for k, v in count_dict.items() if v > 1 and filter_word(k)]
        else:
            # 字粒度
            for sentence in tqdm(sentences):
                chars = list(sentence)
                if self.stop_words:
                    chars = [char for char in chars if char not in self.stop_words and char != ' ']
                tokens.extend(chars)
            # 根据词频过滤一部分频率极低的词/字，不加入词表
            count_dict = Counter(tokens)
            tokens = [k for k, v in count_dict.items() if filter_char(k, remove_sp=self.remove_sp)]
        token2id = dict(zip(tokens, range(1, len(tokens) + 1)))
        id2token = dict(zip(range(1, len(tokens) + 1), tokens))
        # 向生成的词表和标签表中加入[PAD]
        id2token[0] = self.PADDING
        token2id[self.PADDING] = 0
        # 向生成的词表中加入[UNK]
        id2token[len(id2token)] = self.UNKNOWN
        token2id[self.UNKNOWN] = len(id2token)
        # 保存词表及标签表
        with open(self.token_file, 'w', encoding='utf-8') as outfile:
            for idx in id2token:
                outfile.write(id2token[idx] + '\t' + str(idx) + '\n')
        self.vocab_size = len(token2id)
        self.token2id = token2id
        self.id2token = id2token
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

    @staticmethod
    def tokenizer_for_pretrained_model(model_type):
        if model_type == 'Bert':
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(classifier_config['pretrained'])
        elif model_type == 'DistilBert':
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained(classifier_config['pretrained'])
        elif model_type == 'AlBert':
            from transformers import AlbertTokenizer
            tokenizer = AlbertTokenizer.from_pretrained(classifier_config['pretrained'])
        elif model_type == 'Electra':
            from transformers import ElectraTokenizer
            tokenizer = ElectraTokenizer.from_pretrained(classifier_config['pretrained'])
        elif model_type == 'RoBerta':
            from transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained(classifier_config['pretrained'])
        elif model_type == 'XLNet':
            from transformers import XLNetTokenizer
            tokenizer = XLNetTokenizer.from_pretrained(classifier_config['pretrained'])
        else:
            tokenizer = None
        return tokenizer

    def tokenizer_for_sentences(self, sent):
        tokens = []
        sent = self.padding(sent)
        for token in sent:
            if token in self.token2id:
                tokens.append(self.token2id[token])
            else:
                tokens.append(self.token2id[self.UNKNOWN])
        return tokens

    def prepare_w2v_data(self, sentences, labels):
        """
        输出word2vec做embedding之后的X矩阵和y向量
        """
        self.logger.info('loading data...')
        X, y = [], []
        for record in tqdm(zip(sentences, labels)):
            sentence = self.processing_sentence(record[0], self.stop_words)
            tokens = self.tokenizer_for_sentences(sentence)
            label = tf.one_hot(record[1], depth=self.max_label_number)
            X.append(tokens)
            y.append(label)
        return np.array(X), np.array(y, dtype=np.float32)

    def prepare_pretrained_data(self, sentences, labels):
        """
        输出预训练做embedding之后的X矩阵和y向量
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
                sentence = self.processing_sentence(record[0], self.stop_words)
            else:
                sentence = list(record[0])
                if self.stop_words:
                    sentence = [char for char in sentence if char not in self.stop_words and char != ' ']
            label = tf.one_hot(record[1], depth=self.max_label_number)
            tokens = self.tokenizer_for_sentences(sentence)
            X.append(tokens)
            y.append(label)
        return np.array(X), np.array(y, dtype=np.float32)

    def get_dataset(self, df):
        """
        构建Dataset
        """
        df = df.loc[df.label.isin(self.classes)]
        df['label'] = df.label.map(lambda x: self.class_id[x])
        # convert the data in matrix
        if self.classifier in self.support_pretrained_model:
            X, y = self.prepare_pretrained_data(df['sentence'], df['label'])
        else:
            if self.embedding_method == 'word2vec':
                X, y = self.prepare_w2v_data(df['sentence'], df['label'])
            else:
                X, y = self.prepare_data(df['sentence'], df['label'])
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        return dataset

    def prepare_single_sentence(self, sentence):
        """
        把预测的句子转成矩阵和向量
        :param sentence:
        :return:
        """
        if self.classifier in self.support_pretrained_model:
            if len(sentence) > self.max_sequence_length - 2:
                sentence = sentence[:self.max_sequence_length - 2]
                tokens = self.tokenizer.encode(sentence)
            else:
                tokens = self.tokenizer.encode(sentence)
            if len(tokens) < 150:
                tokens += [0 for _ in range(self.max_sequence_length - len(tokens))]
            return np.array([tokens])
        else:
            if self.embedding_method == 'word2vec':
                sentence = self.w2v_util.processing_sentence(sentence, self.stop_words)
                tokens = self.tokenizer_for_sentences(sentence)
                return np.array([tokens])
            else:
                if self.token_level == 'word':
                    sentence = self.w2v_util.processing_sentence(sentence, self.stop_words)
                else:
                    sentence = list(sentence)
                    if self.stop_words:
                        sentence = [char for char in sentence if char not in self.stop_words and char != ' ']
                tokens = self.tokenizer_for_sentences(sentence)
                return np.array([tokens])
