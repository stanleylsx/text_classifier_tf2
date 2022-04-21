# Text Classifier
**公众号文章：[文本分类之Text-CNN/RNN/RCNN算法原理及工程实现](https://mp.weixin.qq.com/s/7fbTt3Ov715ixErYfKR2kA)**  
**公众号文章：[一篇文章带你走进词向量并掌握Word2Vec](https://mp.weixin.qq.com/s/SAEV6WkbkOxzTCvF6GUz_A)**

此仓库是基于Tensorflow2.3的文本分类任务，通过直接配置可支持:  

* **TextCNN/TextRNN/TextRCNN/Transformer/Finetune-Bert基本分类模型的训练** 
* **TextCNN/TextRNN/TextRCNN/Transformer的token可选用词粒度/字粒度** 
* **Word2Vec特征增强后接TextCNN/TextRNN/TextRCNN/Transformer**  
* **支持Attention-TextCNN/TextRNN**  
* **FGM和PGD两种对抗方法的引入训练**  
* **对比学习方法R-drop引入**  
* **支持二分类和多分类，支持FocalLoss**  
* **保存为pb文件可供部署**  
* **项目代码支持交互测试和批量测试**  


## 环境
* python 3.6.7
* tensorflow==2.3.0
* gensim==3.8.3
* jieba==0.42.1
* sklearn==0.0  
* transformers==4.6.1  

其他环境见requirements.txt

## 更新历史
日期| 版本     |描述
:---|:-------|---
2018-12-01| v1.0.0 |初始仓库
2020-10-20| v2.0.0 |重构项目
2020-10-26| v2.1.0 |加入F1、Precise、Recall分类指标,计算方式支持macro、micro、average、binary
2020-11-19| v2.3.0 |加入TextRCNN,加入Attention
2020-11-26| v2.3.1 |加入focal loss用于改善标签分布不平衡的情况
2020-11-19| v2.4.0 |增加每个类别的指标,重构指标计算逻辑
2021-03-02| v2.5.0 |使用Dataset替换自己写的数据加载器来加载数据
2021-03-15| v3.0.0 |支持仅使用TextCNN/TextRCNN进行数据训练(基于词粒度的token,使用随机生成的Embedding层)
2021-03-16| v3.1.0 |支持取用Word2Vec的词向量后接TextCNN/TextRCNN进行数据训练;在log中打印配置
2021-03-17| v3.1.1 |根据词频过滤一部分频率极低的词,不加入词表
2021-03-23| v3.1.3 |加入TextRNN模型,给TextRNN模型加上Attention
2021-03-29| v3.1.5 |增加一个save模块用于保存pb格式的模型文件方便制作tf-severing接口
2021-04-25| v3.1.6 |通过配置可选GPU和CPU进行训练
2021-06-17| v3.2.0 |增加字粒度的模型训练预测
2021-09-27| v3.3.0 |增加测试集的批量测试
2021-11-01| v4.0.0 |增加对抗训练，目前支持FGM和PGD两种方式;增加Bert微调分类训练;更换demo数据集
2021-11-24| v4.2.0 |增加Transformer模型做文本分类、增加对比学习方法r-drop
2022-04-22| v4.3.0 |批量测试打印bad_case以及预测混淆情况、文件夹检查、配置里面不再自己定义标签顺序


## 数据集
部分头条新闻数据集

## 原理
### Word2vec
可以参考我的博客文章[01-NLP介绍和词向量](https://lishouxian.cn/2020/04/06/NLP%E4%BB%8B%E7%BB%8D%E5%92%8C%E8%AF%8D%E5%90%91%E9%87%8F/#WordNet)和[02-词向量第二部分和词义](https://lishouxian.cn/2020/04/13/%E8%AF%8D%E5%90%91%E9%87%8F%E7%AC%AC%E4%BA%8C%E9%83%A8%E5%88%86%E5%92%8C%E8%AF%8D%E4%B9%89/)

### TextCNN
![textcnn](https://img-blog.csdnimg.cn/20201021000109653.png)

### TextRNN
![textrnn](https://img-blog.csdnimg.cn/20210322154656886.png)

### TextRCNN
![textrcnn](https://img-blog.csdnimg.cn/20201107140825534.png)

### Finetune-Bert  
![bert](https://img-blog.csdnimg.cn/ee8c075812ac48b5b2adbfe10d294657.png)

***注(1):这里使用的[transformers](https://github.com/huggingface/transformers)包加载Bert，初次使用的时候会自动下载Bert的模型***   

### Transformer  
从我另外一个[项目](https://github.com/StanleyLsx/text_classification_by_transformer)中集成过来

## 使用
### 配置
在config.py中配置好各个参数，文件中有详细参数说明

### 训练word2vec
在config.py中的mode中改成train_word2vec并运行
```
# [train_classifier, interactive_predict, train_word2vec, save_model, test]
mode = 'train_word2vec'
```

### 训练分类器
配置好下列参数  
```
classifier_config = {
    # 模型选择
    # textcnn/textrnn/textrcnn/Bert/transformer
    'classifier': 'textcnn',
    # 若选择Bert系列微调做分类，请在bert_op指定Bert版本
    'bert_op': 'bert-base-chinese',
    # 训练数据集
    'train_file': 'data/train_dataset.csv',
    # 验证数据集
    'val_file': 'data/val_dataset.csv',
    # 测试数据集
    'test_file': 'data/test_dataset.csv',
    # 引入外部的词嵌入,可选word2vec、Bert
    # word2vec:使用word2vec词向量做特征增强
    # 不填写则随机初始化的Embedding
    'embedding_method': '',
    # token的粒度,token选择字粒度的时候，词嵌入(embedding_method)无效
    # 词粒度:'word'
    # 字粒度:'char'
    'token_level': 'word',
    # 不外接词嵌入的时候需要自定义的向量维度
    'embedding_dim': 300,
    # 存放词表的地方
    'token_file': 'data/word-token2id',
    # 类别和对应的id
    'classes': {'家居': 0, '时尚': 1, '教育': 2, '财经': 3, '时政': 4, '娱乐': 5, '科技': 6, '体育': 7, '游戏': 8, '房产': 9},
    # 模型保存的文件夹
    'checkpoints_dir': 'model/textcnn',
    # 模型保存的名字
    'checkpoint_name': 'textcnn',
    # 使用Textcnn模型时候设定卷集核的个数
    'num_filters': 64,
    # 学习率
    # 微调Bert时建议更小
    'learning_rate': 0.0005,
    # 训练epoch
    'epoch': 100,
    # 最多保存max_to_keep个模型
    'max_to_keep': 1,
    # 每print_per_batch打印
    'print_per_batch': 100,
    # 是否提前结束
    'is_early_stop': True,
    # 是否引入attention
    # 注意:textrcnn不支持
    'use_attention': False,
    # attention大小
    'attention_size': 300,
    'patient': 8,
    'batch_size': 256,
    'max_sequence_length': 300,
    # 遗忘率
    'dropout_rate': 0.5,
    # 隐藏层维度
    # 使用textrcnn、textrnn和transformer中需要设定
    # 使用transformer建议设定为2048
    'hidden_dim': 256,
    # 编码器个数(使用transformer需要设定)
    'encoder_num': 1,
    # 多头注意力的个数(使用transformer需要设定)
    'head_num': 12,
    # 若为二分类则使用binary
    # 多分类使用micro或macro
    'metrics_average': 'micro',
    # 类别样本比例失衡的时候可以考虑使用
    'use_focal_loss': False,
    # 是否使用GAN进行对抗训练
    'use_gan': False,
    # 目前支持FGM和PGD两种方法
    # fgm:Fast Gradient Method
    # pgd:Projected Gradient Descent
    'gan_method': 'pgd',
    # 使用对比学习，不推荐和对抗方法一起使用，效率慢收益不大
    'use_r_drop': False
}
```
配置完参数之后开始训练模型  
```
# [train_classifier, interactive_predict, train_word2vec, test]
mode = 'train_classifier'
```
* 训练结果  

![train_results_textcnn](https://img-blog.csdnimg.cn/949975114b5e46b68f8a019d7d34204e.png)

### 测试
训练好模型直接可以开始测试，可以进行交互测试也可以批量测试  
* 交互测试
```
# [train_classifier, interactive_predict, train_word2vec, test]
mode = 'interactive_predict'  
```
交互测试结果    
![interactive_predict](https://img-blog.csdnimg.cn/433787e1760b45968536b8315ad8e581.png)    

* 批量测试   

在测试数据集配置上填和训练/验证集文件同构的文件地址
```
# 测试数据集
'test_file': 'data/test_dataset.csv',
```
模式设定为测试模式  
```
# [train_classifier, interactive_predict, train_word2vec, test]
mode = 'test'    
```  
批量测试结果    
![batch_test](https://img-blog.csdnimg.cn/bd22c813350449ef937b3a50e1f09322.png) 

## 公众号
相关问题欢迎在公众号反馈：  

![小贤算法屋](https://img-blog.csdnimg.cn/20210427094903895.jpg)


