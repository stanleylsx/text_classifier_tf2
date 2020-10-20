# Comments Classifier
此仓库是基于Tensorflow2.3的评论分类任务，使用Word2vec+TextCNN模型，代码支持二分类和多分类，项目做了个二元情感分类的demo。

## 环境
* python 3.6.7
* tensorflow==2.3.0
* gensim==3.8.3
* jieba==0.42.1
* sklearn==0.0  

其他环境见requirements.txt

## 数据集
我的另外一个爬虫项目[app_comments_spider](https://github.com/StanleyLsx/app_comments_spider)中爬取

## 原理
### Word2vec
可以参考我的博客文章[01-NLP介绍和词向量](https://lishouxian.cn/2020/04/06/NLP%E4%BB%8B%E7%BB%8D%E5%92%8C%E8%AF%8D%E5%90%91%E9%87%8F/#WordNet)和[02-词向量第二部分和词义](https://lishouxian.cn/2020/04/13/%E8%AF%8D%E5%90%91%E9%87%8F%E7%AC%AC%E4%BA%8C%E9%83%A8%E5%88%86%E5%92%8C%E8%AF%8D%E4%B9%89/)    
也可看博客[刘建平Pinard](https://www.cnblogs.com/pinard/p/7160330.html)和文章[技术干货 | 漫谈Word2vec之skip-gram模型](https://mp.weixin.qq.com/s/reT4lAjwo4fHV4ctR9zbxQ?)
### TextCNN
![textcnn](https://img-blog.csdnimg.cn/20201021000109653.png)

## 使用
### 配置
在config.py中配置好各个参数，文件中有详细参数说明

### 训练word2vec
在main.py中的mode中改成train_word2vec并运行
```
# 模式选择
mode = 'train_word2vec'
```

### 训练textcnn
训练好word2vec模型后，开始训练textcnn
```
# 模式选择
mode = 'train_textcnn'
```
* 训练结果  

![train_results](https://img-blog.csdnimg.cn/20201021000109593.png)

### 测试
训练好textcnn可以开始测试
```
# 模式选择
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
