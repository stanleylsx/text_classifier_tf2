import numpy as np
from preprocession import processing_word, get_stop_words
from comment_analysis import build_word_vector
from gensim.models.word2vec import Word2Vec
from sklearn.externals import joblib


# 载入word2vec和svm训练好的模型做预测
def svm_predict(comment):
    n_dim = 300
    svm_clf = joblib.load('svm_model.pkl')
    w2v_model = Word2Vec.load('w2v_model.pkl')
    stop_words_list = get_stop_words()
    processed_comment = processing_word(comment, stop_words_list)
    comment_row = np.array(processed_comment).reshape(1, -1)
    comment_vectors = np.concatenate([build_word_vector(z, n_dim, w2v_model) for z in comment_row])
    predict_result = svm_clf.predict(comment_vectors)
    if int(predict_result) == 0:
        print('差评')
    else:
        print('好评')


svm_predict('很好玩啊')




