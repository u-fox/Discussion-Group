# 新闻自动提取标题
### 1.1 问题背景
过去的研究主要在自动提取摘要，并无很多提取标题的研究，因此可以在文档级摘要研究的基础上研究
### 1.2发展历史
1. 潜在语义分析（LSA）等（启发统计式算法）
2. 递归神经网络（R2N2）自动学习解析树上的排名特征（RNN及其变种，在长文本中表现不佳）
3. 新的神经网络模型（biLSTM 编码器读取输入，使用 LSTM 解码器生成输出，主要贡献是一种新的注意力内策略，和一种新的训练方法，将标准监督式词预测和强化学习结合起来）||（新颖的架构，以两种正交方式增强标准的序列到序列注意力模型…）||Pointer Network框架，指针生成网络。  

（商品推荐）
![](https://image.zhihuishu.com/zhs/ablecommons/demo/202010/64d0e5ac20ae45cbb89256f55b801c4a.png)

商品生成标题
### 1.3研究所需的数据集
Kaggle中中国日报的新闻数据
标题，文本，图片，链接
![](https://image.zhihuishu.com/zhs/ablecommons/demo/202010/98fb3ab214a74c8da3ee5e97b296db55.png)

### 2.1方法方向
**2.11 抽取式**  

- 将问题视为序列标注的问题
- 各项文献表示并不适用于标题

**2.12 生成式**  

基于序列到序列（Sequence-to-Sequence）模型  

### 2.2 难点与问题
1. 内容或细节不准确，造成标题无用
2. 从摘要过度到标题往往存在过长的问题
3. 学习特定得到的标题风格能否迁移
4. 整体中提取，但生成的难以构成一个有意义的整体

### 2.3 最基本seq2seq效果
![](https://image.zhihuishu.com/zhs/ablecommons/demo/202010/aa2aa925281e4f6282f8054f90a85423.png)

### 3.1未来发展
注意力机制，端到端的新神经网络框架  

**Reference**
- [1] Oriol Vinyals, Meire Fortunato, and Navdeep Jaitly. 2015. Pointer Networks. In Proceedings of NIPS, C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett (Eds.). Curran Associates, Inc., 2692–2700.
- [2] Katja Filippova, Enrique Alfonseca, Carlos A. Colmenares, Lukasz Kaiser, and Oriol Vinyals. 2015. Sentence Compression by Deletion with LSTMs. In Proceedings of EMNLP. Association for Computational Linguistics, Lisbon, Portugal, 360–368.
- [3] Hongyan Jing. 2002. Using Hidden Markov Modeling to Decompose Human-written Summaries. Comput. Linguist. 28, 4 (Dec. 2002), 527–543.
- [4]How can catchy titles be generated without loss of informativeness?
- [5] 基于深度学习的标题生成方法综述蒋敏
