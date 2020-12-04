# -*- coding: utf-8 -*-
# @Time : 2020/11/19 12:12 下午 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : sentence2vec.py 
# @Software: PyCharm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from engines.utils.word2vec import Word2VecUtils
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm


class Sentence2VecUtils:
    def __init__(self, logger):
        self.w2v_utils = Word2VecUtils(logger)
        self.w2v_model = Word2Vec.load(self.w2v_utils.model_path)
        self.pca_vec_path = './pca_u.npy'
        self.count_num = 0
        for key, value in self.w2v_model.wv.vocab.items():
            self.count_num += value.count
        self.logger = logger
        self.a = 1e-3

    def calculate_weight(self, sentence):
        vs = np.zeros(self.w2v_utils.dim)  # add all word2vec values into one vector for the sentence
        sentence_length = len(sentence)
        for word in sentence:
            if word in self.w2v_model.wv.vocab:
                p_w = self.w2v_model.wv.vocab[word].count / self.count_num
                a_value = self.a / (self.a + p_w)  # smooth inverse frequency, SIF
                vs = np.add(vs, np.multiply(a_value, np.array(self.w2v_model[word])))  # vs += sif * word_vector
        if sentence_length != 0:
            vs = np.divide(vs, sentence_length)  # weighted average
        else:
            vs = None
        return vs

    def train_pca(self):
        sentence_set = []
        # 切词
        stop_words = self.w2v_utils.get_stop_words()
        train_df = pd.read_csv(self.w2v_utils.train_data, encoding='utf-8')
        self.logger.info('Cut sentence...')
        train_df['sentence'] = train_df.sentence.apply(self.w2v_utils.processing_sentence, args=(stop_words,))
        # 删掉缺失的行
        train_df.dropna(inplace=True)
        sentence_list = train_df.sentence.to_list()
        for sentence in tqdm(sentence_list):
            vs = self.calculate_weight(sentence)
            if vs is not None:
                sentence_set.append(vs)  # add to our existing re-calculated set of sentences
            else:
                continue

        # calculate PCA of this sentence set
        pca = PCA(n_components=self.w2v_utils.dim)
        pca.fit(np.array(sentence_set))
        u = pca.components_[0]  # the PCA vector
        u = np.multiply(u, np.transpose(u))  # u x uT
        np.save(self.pca_vec_path, u)

    def get_sif_vector(self, sentence, u):
        vs = self.calculate_weight(sentence)
        sub = np.multiply(u, vs)
        result_vec = np.subtract(vs, sub)
        return result_vec
