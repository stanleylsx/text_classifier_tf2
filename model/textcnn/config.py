# -*- coding: utf-8 -*-
# @Time : 2020/10/20 11:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py 
# @Software: PyCharm


# [train_classifier, interactive_predict, train_word2vec, save_model]
mode = 'interactive_predict'

word2vec_config = {
    'stop_words': 'data/w2v_data/stop_words.txt',  # 停用词(可为空)
    'train_data': 'data/w2v_data/comments_data.csv',  # 词向量训练用的数据
    'model_dir': 'model/w2v_model',  # 词向量模型的保存文件夹
    'model_name': 'w2v_model.pkl',  # 词向量模型名
    'word2vec_dim': 300,  # 词向量维度
}

CUDA_VISIBLE_DEVICES = 0
# int, -1:CPU, [0,]:GPU
# coincides with tf.CUDA_VISIBLE_DEVICES

classifier_config = {
    # 模型选择
    'classifier': 'textcnn',
    # 训练数据集
    'train_file': 'data/data/train_data.csv',
    # token粒度,token选择字粒度的时候，词嵌入无效
    # 词粒度:'word'
    # 字粒度:'char'
    'token_level': 'word',
    # 引入外部的词嵌入,可选word2vec、Bert
    # 此处只使用Bert Embedding,不对其做预训练
    # None:使用随机初始化的Embedding
    'embedding_method': None,
    # 不外接词向量的时候需要自定义的向量维度
    'embedding_dim': 300,
    # 存放词表的地方
    'token_file': 'data/data/token2id',
    # 验证数据集
    'dev_file': 'data/data/dev_data.csv',
    # 类别和对应的id
    'classes': {'negative': 0, 'positive': 1},
    # 模型保存的文件夹
    'checkpoints_dir': 'model/textcnn',
    # 模型保存的名字
    'checkpoint_name': 'textcnn',
    # 卷集核的个数
    'num_filters': 64,
    # 学习率
    'learning_rate': 0.001,
    # 训练epoch
    'epoch': 30,
    # 最多保存max_to_keep个模型
    'max_to_keep': 1,
    # 每print_per_batch打印
    'print_per_batch': 20,
    # 是否提前结束
    'is_early_stop': True,
    # 是否引入attention
    # 注意:textrcnn不支持
    'use_attention': False,
    # attention大小
    'attention_size': 300,
    'patient': 8,
    'batch_size': 64,
    'max_sequence_length': 150,
    # 遗忘率
    'dropout_rate': 0.5,
    # 隐藏层维度
    # 使用textrcnn中需要设定
    'hidden_dim': 200,
    # 若为二分类则使用binary
    # 多分类使用micro或macro
    'metrics_average': 'binary',
    # 类别样本比例失衡的时候可以考虑使用
    'use_focal_loss': False
}
