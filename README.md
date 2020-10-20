# 10月25日前将会升级代码，从word2vec+svm构建到word2vec+textcnn

# comments_analysis
所有的数据均来源于项目[app_comments_spider](https://github.com/StanleyLsx/app_comments_spider)

## 环境与配置

## word2vec简介

## 原理与训练  

## 预测
&emsp;&emsp;最后，comment_predict.py文件用来做评论预测，对于新的评论，它也会对这些句子进行预处理(分词、去停用词)，然后向量化句子中的每个词汇，求句子的词汇向量平均值，然后直接把处理好的向量送进svm进行预测。
