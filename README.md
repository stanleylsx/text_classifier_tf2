# Text Classifier
此仓库是基于Tensorflow2.3的文本分类任务，通过直接配置可分别支持:  

* 随机初始Word Embedding + TextCNN  
* 随机初始Word Embedding + Attention + TextCNN  
* 随机初始Word Embedding + TextRNN  
* 随机初始Word Embedding + Attention + TextRNN   
* 随机初始Word Embedding + TextRCNN   
* Word2Vec + TextCNN   
* Word2Vec + Attention + TextCNN  
* Word2Vec + TextRNN  
* Word2Vec + Attention + TextRNN  
* Word2Vec + TextRCNN  
* Bert Embedding(没有微调,直接取向量) + TextCNN    
* Bert Embedding(没有微调,直接取向量) + TextRCNN    
* Bert Embedding(没有微调,直接取向量) + TextRNN    

代码支持二分类和多分类，此项目基于爬取的游戏评论做了个二元的情感分类作为demo。  

## 环境
* python 3.6.7
* tensorflow==2.3.0
* gensim==3.8.3
* jieba==0.42.1
* sklearn==0.0  

其他环境见requirements.txt

## 更新历史
日期|版本|描述
:---|:---|---
2018-12-01|v1.0.0|初始仓库
2020-10-20|v2.0.0|重构项目
2020-10-26|v2.1.0|加入F1、Precise、Recall分类指标,计算方式支持macro、micro、average、binary
2020-11-06|v2.2.0|加入TextRCNN
2020-11-19|v2.3.0|加入Attention
2020-11-26|v2.3.1|加入focal loss用于改善标签分布不平衡的情况
2020-11-19|v2.4.0|增加每个类别的指标,重构指标计算逻辑
2021-03-02|v2.5.0|使用Dataset替换自己写的数据加载器来加载数据
2021-03-15|v3.0.0|支持仅使用TextCNN/TextRCNN进行数据训练(基于词粒度的token,使用随机生成的Embedding层)
2021-03-16|v3.1.0|支持取用Bert的编码后接TextCNN/TextRCNN进行数据训练(此项目Bert不支持预训练);在log中打印配置
2021-03-17|v3.1.1|根据词频过滤一部分频率极低的词,不加入词表
2021-03-22|v3.1.2|加入TextRNN模型

## 数据集
我的另外一个爬虫项目[app_comments_spider](https://github.com/StanleyLsx/app_comments_spider)中爬取

## 原理
### Word2vec
可以参考我的博客文章[01-NLP介绍和词向量](https://lishouxian.cn/2020/04/06/NLP%E4%BB%8B%E7%BB%8D%E5%92%8C%E8%AF%8D%E5%90%91%E9%87%8F/#WordNet)和[02-词向量第二部分和词义](https://lishouxian.cn/2020/04/13/%E8%AF%8D%E5%90%91%E9%87%8F%E7%AC%AC%E4%BA%8C%E9%83%A8%E5%88%86%E5%92%8C%E8%AF%8D%E4%B9%89/)    
也可看博客[刘建平Pinard](https://www.cnblogs.com/pinard/p/7160330.html)和文章[技术干货 | 漫谈Word2vec之skip-gram模型](https://mp.weixin.qq.com/s/reT4lAjwo4fHV4ctR9zbxQ?)

### TextCNN
![textcnn](https://img-blog.csdnimg.cn/20201021000109653.png)

### TextRNN
![textrnn](https://img-blog.csdnimg.cn/20210322154656886.png)

### TextRCNN
![textrcnn](https://img-blog.csdnimg.cn/20201107140825534.png)


## 使用
### 配置
在config.py中配置好各个参数，文件中有详细参数说明

### 训练word2vec
在config.py中的mode中改成train_word2vec并运行
```
# [train_classifier, interactive_predict, train_word2vec]
mode = 'train_word2vec'
```

### 训练分类器
配置好下列参数  
```
classifier_config = {
    # 模型选择
    'classifier': 'textcnn',
    # 训练数据集
    'train_file': 'data/data/train_data.csv',
    # 引入外部的词嵌入,可选word2vec、Bert
    # 此处只使用Bert Embedding,不对其做预训练
    # None:使用随机初始化的Embedding
    'embedding_method': 'Bert',
    # 不外接词向量的时候需要自定义的向量维度
    'embedding_dim': 300,
    # 存放词表的地方
    'token_file': 'data/data/token2id',
    # 验证数据集
    'dev_file': 'data/data/dev_data.csv',
    # 类别和对应的id
    'classes': {'negative': 0, 'positive': 1},
    # 模型保存的文件夹
    'checkpoints_dir': 'model/bert_textcnn',
    # 模型保存的名字
    'checkpoint_name': 'bert_textcnn',
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
    'attention_dim': 300,
    'patient': 8,
    'batch_size': 64,
    'max_sequence_length': 150,
    # 遗忘率
    'droupout_rate': 0.5,
    # 隐藏层维度
    # 使用textrcnn中需要设定
    'hidden_dim': 200,
    # 若为二分类则使用binary
    # 多分类使用micro或macro
    'metrics_average': 'binary',
    # 类别样本比例失衡的时候可以考虑使用
    'use_focal_loss': False
}
```
配置完参数之后开始训练模型  
```
# [train_classifier, interactive_predict, train_word2vec]
mode = 'train_classifier'
```
* textcnn训练结果  

![train_results_textcnn](https://img-blog.csdnimg.cn/2020110713592572.png)

* att-textcnn训练结果  

![train_results_att-textcnn](https://img-blog.csdnimg.cn/20201119115846656.png)

* textrcnn训练结果  

![train_results_textrcnn](https://img-blog.csdnimg.cn/20201107140248442.png)

### 测试
训练好textcnn可以开始测试
```
# [train_classifier, interactive_predict, train_word2vec]
mode = 'interactive_predict'
```
* 交互测试结果  

![test](https://img-blog.csdnimg.cn/20201021000109568.png)

## 参考
* [app_comments_spider](https://github.com/StanleyLsx/app_comments_spider)
* [01-NLP介绍和词向量](https://lishouxian.cn/2020/04/06/NLP%E4%BB%8B%E7%BB%8D%E5%92%8C%E8%AF%8D%E5%90%91%E9%87%8F/#WordNet)
* [02-词向量第二部分和词义](https://lishouxian.cn/2020/04/13/%E8%AF%8D%E5%90%91%E9%87%8F%E7%AC%AC%E4%BA%8C%E9%83%A8%E5%88%86%E5%92%8C%E8%AF%8D%E4%B9%89/) 
* [刘建平Pinard](https://www.cnblogs.com/pinard/p/7160330.html)
* [技术干货 | 漫谈Word2vec之skip-gram模型](https://mp.weixin.qq.com/s/reT4lAjwo4fHV4ctR9zbxQ?)
* [Recurrent Convolutional Neural Networks for Text Classification](http://zhengyima.com/my/pdfs/Textrcnn.pdf)

