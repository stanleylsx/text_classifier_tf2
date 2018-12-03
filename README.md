# comments_analysis
&emsp;&emsp;基于word2vec的评论情感分类器，所有的数据均来源于项目[app_comments_spider](https://github.com/StanleyLsx/app_comments_spider)，通过word2vec对带有评分的句子的词向量进行学习，然后去预测其他的不带评分的评论，进一步的，可以找出评论中的意见与建议等。
## 环境与配置
&emsp;&emsp;python为3.6版本，所需的python相关包在requirements下，主要有jiaba、gensim、sklearn等。在进行训练前，需要将数据库的数据导入的excel中，过程由database2xls.py文件完成，相应的，需要在该文件的
`comment_db = DataBase('', '', '', '')`
中填好数据库地址、账号、密码和数据库，由于appstore和taptap都是五星制的评价，所以git项目中为了简单起见，将三星及三星以下做为差评，三星以上作为好评，并将数据整理到comment.xls这个excel中用于二分类问题的预测。
## word2vec简介
&emsp;&emsp;word2vec是google在2013年推出的一个NLP工具，它的特点是将词向量化，这样词与词之间就可以定量的去度量他们之间的关系，挖掘词之间的联系，word2vec有两种模型的实现，分别是CBOW和Skip-Gram模型，前者是通过上下文词向量推断中间的某个词向量，后者是推断特定词对应的上下文词向量，本质上都是基于DNN神经网络的神经网络的相关算法基础可以参考[这里](https://github.com/StanleyLsx/machinelearning#3)，这里讲了DNN、CNN、RNN的原理和相应的算法。word2vec的官方文档可以参考[这里](https://radimrehurek.com/gensim/index.html)，讲得比较好的还有刘建平的[博客](https://www.cnblogs.com/pinard/p/7160330.html)。
## 原理与训练
&emsp;&emsp;关于中文nlp处理的步骤一般是:  
* 分析需求，预测哪个列?二分类还是多分类?需要什么样的结果?  
* 获得数据，一般可以直接来源于数据库，也可以间接通过爬虫实现，[app_comments_spider](https://github.com/StanleyLsx/app_comments_spider)便是本文中的数据来源。  
* 文本预处理，一般先使用jiaba这个工具对句子或者文本进行分词，jieba默认采用的是HMM模型分词，[这里](https://github.com/fxsjy/jieba)是项目地址。划分出来的词一般还要去掉停用词，项目中的stop_words.txt放的就是一些停用词。  
* 特征处理，在本文中即词汇的向量化，将文本预处理之后的词汇利用word2vec进行向量化，有的热点文章还需要使用TF-IDF对文本进行预处理，TF-IDF即词频-逆文本频率，对于高频词汇降低其重要性，对出现频率少的词提高其重要性。  
* 建立模型，通过向量化后的词汇和相应的输出训练svm、逻辑回归这些分类器，调参获得更好的准确率和r_score，最后拿着建立好的模型去预测。  

&emsp;&emsp;项目的preprocession.py文件将获得评论进行分词、去停用词和训练数据集的划分。comment_analysis.py会将训练数据集和测试数据集的评论中的词汇通过word2vec进行向量化，每个词汇有300个维度，然后对每个句子求它所含有的词汇向量的平均值并归一化，并把训练好的词向量保存为w2v_model.pkl。最后，用训练数据集中句子的向量平均值去训练svm数据，通过预测结果和测试数据集的对比得到score，训练数据集够量的时候，预测准确度接近90%，把训练好的模型保存为svm_model.pkl以备后面直接载入进行预测调用。
## 预测
&emsp;&emsp;最后，comment_predict.py文件用来做评论预测，对于新的评论，它也会对这些句子进行预处理(分词、去停用词)，然后向量化句子中的每个词汇，求句子的词汇向量平均值，然后直接把处理好的向量送进svm进行预测。
