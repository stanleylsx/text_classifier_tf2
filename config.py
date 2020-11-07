# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py 
# @Software: PyCharm


# [train_classifier, interactive_predict, train_word2vec]
mode = 'interactive_predict'

word2vec_config = {
    'stop_words': 'data/w2v_data/stop_words.txt',  # 停用词(可为空)
    'train_data': 'data/w2v_data/comments_data.csv',  # 词向量训练用的数据
    'word2vec_model': 'model/w2v_model/w2v_model.pkl',  # 词向量模型的保存地址
    'word2vec_dim': 300,  # 词向量维度
}

classifier_config = {
    'model': 'textcnn',  # 模型选择
    'train_file': 'data/data/train_data.csv',  # 训练数据集
    'dev_file': 'data/data/dev_data.csv',  # 验证数据集
    'classes': {'negative': 0, 'positive': 1},  # 类别和对应的id
    'checkpoints_dir': 'model/textrcnn_model',  # 模型保存的文件夹
    'checkpoint_name': 'textrcnn_model',  # 模型保存的名字
    'num_filters': 64,  # 卷集核的个数
    'learning_rate': 0.001,  # 学习率
    'epoch': 30,  # 训练epoch
    'max_to_keep': 1,  # 最多保存max_to_keep个模型
    'print_per_batch': 20,  # 每print_per_batch打印
    'is_early_stop': True,  # 是否提前结束
    'patient': 8,
    'batch_size': 64,
    'max_sequence_length': 150,
    'droupout_rate': 0.3,  # 遗忘率
    'hidden_dim': 200,  # 隐藏层维度
    'metrics_average': 'binary',  # 若为二分类则使用binary，多分类使用micro或macro
}
