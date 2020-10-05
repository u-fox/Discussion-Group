# WEEK2:数据集

## 制作数据集

### 第一篇文章：[Ancient-Modern Chinese Translation with a New Large Training Dataset](https://arxiv.org/pdf/1808.03738.pdf)

这篇文章就是讲如何**制作古今汉语的大规模并行语料库**，并用几个机器翻译的模型上都跑了一遍

#### 古今汉语的一些问题

1. 古今汉语的**语法**有很大不同
2. 古今汉语**音节**不同，现代汉语多为两个字表意、古代汉语为一个字表意
3. 古今汉语为**一对多**（古今汉语更多的是**多对多**）
4. 古今汉语语言**汉字共现**现象

#### 获取翻译实例

一般来说，获取翻译实例的最好办法是**双语文本对齐**

##### 文本对齐

[From the Rosetta stone to the information society](http://proxy.qsclub.cn/index.php?_proxurl=aHR0cDovL2NpdGVzZWVyeC5pc3QucHN1LmVkdS92aWV3ZG9jL2Rvd25sb2FkP2RvaT0xMC4xLjEuMTMuNTI0MCZyZXA9cmVwMSZ0eXBlPXBkZg%3D%3D&_proxfl=1eb)

这篇文章主要介绍了平行文本的处理与使用。分为三个部分：

1. **在句子、词语（不同层次）上进行文本对齐的技术与方法**
2. 平行文本在翻译等领域的应用
3. **目前可用的语料库和对语句对齐结果的评价方法**

> Q：什么是平行文本
>
> A：不同的语言，表示一种意思

###### 自动平行文本对齐方法（bitexts）

一般分为四个维度：**段落、句子、单词和短语**

一些基本假设：

1. 为了使译文中的句子对应，其中的单词也必须对应【Kay & Röscheisen (1988, 1993)】

> 即所有产生的词语间的映射需要来自文本本身，**词语的成功对齐可以导致语句、段落的对齐**。

[(PDF) Text-Translation Alignment - ResearchGate](https://www.researchgate.net/publication/220355417_Text-Translation_Alignment)

> 上方文章提出了一种**仅基于内部证据**的文本与翻译对齐算法

#### 古今汉语的对齐

一般结合使用词法方法与统计学方法

[Sentence Alignment for Ancient and Modern Chinese Parallel Corpus](https://www.researchgate.net/publication/287093608_Sentence_Alignment_for_Ancient_and_Modern_Chinese_Parallel_Corpus)中提出一个对数线性模型进行自动对齐，以句子长度、对齐方式、共现汉字为特征。为纯统计学方法。

|   特征   |                特点                |               备注               |
| :------: | :--------------------------------: | :------------------------------: |
| 句子长度 | 长句往往翻译之后也是长句，短句亦然 | 句长一般为字符个数、词语个数两种 |

[Automatic Translating Between Ancient Chinese and Contemporary Chinese with Limited](https://arxiv.org/pdf/1803.01557.pdf)中提出了一个基于最长公共子序列的简单古今汉语句子对齐方法，属于无监督模型。为词法方法+统计学方法。考虑了古今汉语存在大量共有字符的情况。

他们借助[Get To The Point: Summarization with Pointer-Generator Networks](http://proxy.qsclub.cn/index.php?_proxurl=aHR0cHM6Ly93d3cuYWNsd2ViLm9yZy9hbnRob2xvZ3kvUDE3LTEwOTkucGRm)中指针生成器模型的复制机制处理共有字符这种情况，开发具有复制机制和局部注意力机制的seq2seq模型

[Ancient-Modern Chinese Translation with a New Large Training Dataset](https://arxiv.org/pdf/1808.03738.pdf)中将句子划分更细

### 第二篇文章：[Chinese Ancient-Modern Sentence Alignment](http://proxy.qsclub.cn/index.php?_proxurl=aHR0cHM6Ly93d3cucmVzZWFyY2hnYXRlLm5ldC9wdWJsaWNhdGlvbi8yMjA4NTkyNTZfQ2hpbmVzZV9BbmNpZW50LU1vZGVybl9TZW50ZW5jZV9BbGlnbm1lbnQ%3D&_proxfl=1eb)

本文主要研究n-m对齐模式（多对多）

### 搭建

[A Program for Aligning Sentences in Bilingual Corpora](https://www.aclweb.org/anthology/J93-1004.pdf)
