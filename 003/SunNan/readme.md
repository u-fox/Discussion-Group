# 1 研究所需的数据集
（几十个G过大的数据集都需要发送邮件取得硬盘数据）


## 1.1 英文
### 1.美国各大新闻出版社新闻数据集[1]
14万条
内容较为全面标准
### 2.DUC数据集[7]
较为经典的新闻语料库，被专门用来训练nlp task
也是一些paper新闻标题训练用的语料库
缺点在于数据只在百量级
被论文[6]引用训练

## 1.2 中文
### 1.新闻专栏数据集
1万条
标题风格：通俗易懂，大众化
### 2.搜狐分类新闻数据集（搜狗实验室）
大小为65G，数据量极大。
来自搜狐新闻2012年6月—7月期间国内，国际，体育，社会，娱乐等18个频道的新闻数据
数据格式为:
```xml
<doc>
<url>页面URL</url>
<docno>页面ID</docno>
<contenttitle>页面标题</contenttitle>
<content>页面内容</content>
</doc>
```

### 1.3 全网新闻数据集（搜狗实验室）
700兆左右


# 2 研究现状综述（最新综述论文[2]）
## 2.1 抽取式自动文本摘要研究现状
第一种是基于统计与各种机器学习方法，对文本提取建模[4]。  
第二种是基于图排序的自动文本摘要将文本划分成若干个段落的集合或者是若干个句子的集合，一个集合对应一个图顶点，集合与集合之间存在的关系对应结构边，最后通过例如 PageRank、HITS、TextRank 等图排序算法计算各个图顶点的得分，根据得分选择重要句子生成摘要[3]  
## 2.2 生成式自动文本摘要研究现状
针对生成式文本摘要和标题生成任务的序列到序列模型，一般由嵌入层、编码层和解码层以及训练策略组成。在嵌入层，可以对词向量进行随机化，大多数工作使用 Mikolov[18]提出的Word2vec 和 Pennington[19]提出的 Glove 词向量工具对文本向量化。Nallapati 等人[20]在词嵌入层将词性，命名实体标签，单词的 TF 和 IDF 等文本特征融合在词向量中，让单词具有多个维度的意义。  

Wang 等人[21]则在词向量中加入位置向量作为最终的文本向量表示，丰富词向量的含义。 在编码层，多数模型使用双向长短期记忆网络（Bidirectional  Long  Short-Term Memory，Bi-LSTM）和门控循环单元（Gated Recurrent Unit，GRU）循环神经网络对原文中的每个词进行编码，使得每个词都具有上下文信息。

Zhou 等人[22]在编码器中引入选择门网络，从源文本中提取重要信息，去除冗余信息提高摘要质量。

Zeng等人[23]考虑到人写摘要时会多次阅读一篇文章，提出一种再读编码器，读取源文本两次，得到比较合理的语义向量。而 Chopra[24]和 Wang[21]等人使用卷积神经网络（Convolutional Neural Networks，CNN）进行文本编码，去除了 LSTM 和 GRU 的循环特性，实现了文本编码时的并行性，从而提升了模型训练速度。 

在解码层，一般模型融合可以选择到原始文本重要信息的注意力机制解码生成摘要或标题，最后利用优化函数进行模型的训练。Han 等人[25]提出一种多样性的波束搜索算法，生成多样的候选序列，促进解码生成信息丰富的摘要。

针对训练策略，Ayana 等人[26]提出将文本摘要的评价指标融合到优化目标内，使用最小风险训练方法训练模型。Paulu 等人[27]在传统的损失函数中加入强化学习损失函数，用以解决传统方法带来的暴露偏差和输出结果的不灵活性等问题。

使用序列到序列模型易产生重复和无法处理词汇表外（Out-of-vocabulary，OOV）等问题，针对这些问题，See 等人[28]提出使用指针网络和覆盖机制解决；Lin 等人[29]在编码器中加入卷积门控单元进行源文本的全局编码，解决输出产生的重复单词或短语问题。 由于序列到序列模型的出现及其特性，生成式摘要得到了极大地发展。标题是对整篇文本的核心提炼，更需要理解文本后得出中心内容，研究者们多选择生成式文本摘要的方法进行文本标题生成[30,31]。

论文[2]中尝试了基于句子级LSTM编码的文本表示方法以及融合词汇语言特征和多头注意力的方法

目前主流论文都是用上述的深度学习解码编码模型，并逐一尝试各种组件
 
   
清华大学论文[6]中指出，影响效果的几大影响因素：Linguistic Feature，Encoder Choice，Decoder Choice，Training Strategy，Output Length，Corpus.
特别指出，语料库对不同系统的训练结果有着不同的影响

主要评价指标用ROUGE

# Reference

- [1] https://www.kaggle.com/snapcrack/all-the-news
- [2] https://kreader.cnki.net/Kreader/CatalogViewPage.aspx?dbCode=cdmd&filename=1020626999.nh&tablename=CMFDTEMP&compose=&first=1&uid=WEEvREcwSlJHSldSdmVqMDh6aSs3RStlTHhHanF4eUcxdkZPVDhmVnFuVT0=$9A4hF_YAuvQ5obgVAqNKPCYcEjKensW4IQMovwHtwkF4VYPoHbKxJw!!
- [3] Mihalcea R. Graph-based ranking algorithms for sentence extraction, applied to text summarization[C]//Proceedings  of  the  ACL  Interactive  Poster  and  Demonstration Sessions. 2004: 170-173.   
- [4]How can catchy titles be generated without loss of informativeness? 
- [6] Recent Advances on Neural Headline Generation
- [7] https://duc.nist.gov/duc2004/ 
- [15] Conroy  J  M,  O'leary  D  P.  Text  summarization  via  hidden  markov models[C]//Proceedings of the 24th annual international ACM SIGIR conference on Research and development in information retrieval. 2001: 406-407.   
- [16] Mihalcea R. Graph-based ranking algorithms for sentence extraction, applied to text summarization[C]//Proceedings  of  the  ACL  Interactive  Poster  and  Demonstration Sessions. 2004: 170-173.   
- [17] Sutskever  I  ,  Vinyals  O  ,  Le  Q  V  .  Sequence  to  Sequence  Learning  with  Neural Networks[C]//Advances in neural information processing systems,2014: 3104-3112. 
- [18] Mikolov  T,  Sutskever  I,  Chen  K,  et  al.  Distributed  representations  of  words  and phrases and their compositionality[J]. 2013, 26:3111-3119. 
- [19] Pennington,  J.,  Socher,  R.,  Manning,  C.  Glove:  Global  vectors  for  word representation[C]//Proceedings  of  the  2014  conference  on  empirical  methods  in natural language processing, 2014:1532-1543.
- [20] Nallapati  R,  Zhou  B,  Santos  C  N  D,  et  al.  Abstractive  Text  Summarization  Using Sequence-to-Sequence RNNs and Beyond[C]//Proceedings of Computational Natural Language Learning, 2016: 280-290. 
- [21] Wang  L,  Yao  J,  Tao  Y,  et  al.   A  Reinforced  Topic-Aware  Convolutional Sequence-to-Sequence  Model  for  Abstractive  Text  Summarization[C]//  Proceedings of the 27th International Joint Conference on Artificial Intelligence, 2018: 4453-4460.
- [22] Zhou  Q,  Yang  N,  Wei  F,  et  al.  Selective  Encoding  for  Abstractive  Sentence Summarization[C]//Proceedings  of  the  55th Annual  Meeting  of  the Association  for Computational Linguistics, 2017: 1095-1104.
- [23] Zeng W, Luo W, Fidler S, et al. Efficient summarization with read-again and copy mechanism[J]. arXiv preprint arXiv:1611.03382, 2016.
- [24] Chopra S, Auli M, Rush A M. Abstractive Sentence Summarization with Attentive Recurrent  Neural  Networks[C]//Proceedings  of  the  North  American  Chapter  of  the Association for Computational Linguistics, 2016: 93-98
- [25] Han X W, Zheng H T, Chen J Y, et al. Diverse Decoding for Abstractive Document Summarization[J]. Applied Sciences, 2019, 9(3): 386. 参考文献 37
- [26] Ayana,  Shen  S,  Liu  Z,  et  al.  Neural  Headline  Generation  with  Minimum  Risk Training[J]. arXiv preprint arXiv:1604.01904, 2016
- [27] Paulus  R,  Xiong  C,  Socher  R.  A  Deep  Reinforced  Model  for  Abstractive Summarization[J]. arXiv preprint arXiv:1705.04304, 2017
- [28] See  A,  Liu  P  J,  Manning  C  D.  Get  To  The  Point:  Summarization  with Pointer-Generator  Networks[C]//Proceedings  of  the  55th  Annual  Meeting  of  the Association for Computational Linguistics, 2017: 1073-1083
- [29] Lin  J,  Sun  X,  Ma  S,  et  al.  Global  encoding  for  abstractive summarization[C]//Proceedings  of  the  56th  Annual  Meeting  of  the  Association  for Computational Linguistics, 2018: 163–169
- [30] Hayashi  Y  ,  Yanagimoto  H  .  Headline  Generation  with  Recurrent  Neural Network[M]// New Trends in E-service and Smart Computing. 2018
- [31] Ayana, Shen S Q , Lin Y K , et al. Recent Advances on Neural Headline Generation[J]. Journal of Computer Science and Technology, 2017, 32(4):768-784.


