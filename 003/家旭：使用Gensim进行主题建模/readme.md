# 说明
本周对此任务的示例文件做了分析，了解学习了其中提到的一些模型概念及库：
- [LDA潜在狄利克雷分布模型](https://blog.csdn.net/aws3217150/article/details/53840029)
- [Gensim基础——语料库与向量空间](https://blog.csdn.net/questionfish/article/details/46739207)
- [tf-idf](https://blog.csdn.net/zrc199021/article/details/53728499)
- [词袋模型和词向量模型](https://www.cnblogs.com/chenyusheng0803/p/10978883.html)
- [词形还原（Lemmatization）](https://www.jianshu.com/p/79255fe0c5b5)
- [spacy分词，命名实体识别，词性识别等](https://www.jianshu.com/p/e6b3565e159d)
- [Word2Vec-知其然知其所以然](https://www.zybuluo.com/Dounm/note/591752)
- [GuidedLDA详解](https://zhuanlan.zhihu.com/p/213841493)

并参照文章[Topic Modeling with Gensim](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python)对代码做出注释。

# 使用Gensim进行主题建模（[Topic Modeling with Gensim](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python)）


主题建模是一种从大量文本中提取隐藏主题的技术。  

Latent Dirichlet Allocation（LDA）是一种流行的主题建模算法，在Python的Gensim包中具有出色的实现。  

然而，挑战在于**如何提取清晰的(clear)，分离的(segregated)和有意义(meaningful)的高质量主题** 。这在很大程度上取决于两点:

- 文本预处理的质量
- 找到最佳主题数量的策略

本例试图解决这两个问题。

## 0.目录
1. 简介
2. 先决条件--下载nltk停用词和spacy模型
3. 导入包
4. LDA做什么？
5. 准备好停用词
6. 导入数据
7. DataFrame转list
8. 标记单词和清理文本
9. 创建Bigram和Trigram模型
10. 删除停用词，制作Bigrams和词形还原
11. 创建主题建模所需的词典和语料库
12. 构建主题模型
13. 查看LDA模型中的主题
14. 计算模型复杂度和相干性分数
15. topics-keywords可视化
16. 构建LDA Mallet模型
17. 如何找到LDA的最佳主题数量？
18. 在每个句子中找到主要话题
19. 找到每个主题最具代表性的文档
20. 将主题分发给各文档
21. 总结

## 1.简介(Introduction)
自然语言处理的主要应用之一是**从大量文本中自动提取人们正在讨论的主题**。大量文本的一些示例可以是来自社交媒体的馈送，酒店的客户评论，电影等，用户反馈，新闻报道，客户投诉的电子邮件等。

了解人们在谈论什么并理解他们的问题和意见对于企业，管理者和政治活动来说非常有价值。并且很难人工阅读如此大数据量的文本并识别主题。

因此，需要一种自动算法，该算法可以读取文本文档并自动输出所讨论的主题。

我将使用Gensim包中的Latent Dirichlet Allocation（LDA）以及Mallet的实现（通过Gensim）。Mallet有效地实现了LDA。众所周知，它可以更快地运行并提供更好的主题隔离。

我们还将提取每个主题的数量和百分比贡献，以了解主题的重要性。

让我们开始！

## 2.先决条件(Prerequisites) - 下载nltk停用词和spacy模型
我们需要来自NLTK的`stopwords`和spacy的`en模型`进行文本预处理。稍后，我们将使用spacy模型进行词形还原。

词形还原只不过是将一个词转换为词根。例如：“machines”这个词的lemma是“machines”。同样，'walking'->'walk'，'mice'->'mouse'等等。

## 3.导入包（Import Packages）
在本例中使用的核心包是`re`，`gensim`，`spacy`和`pyLDAvis`。  

除此之外，我们还将使用`matplotlib`，`numpy`以及`pandas`数据处理和可视化。让我们导入它们。


```python
import re
import json
from IPython.core.display import display, HTML, clear_output
import html
# glob文件名模式匹配，是python自带的一个文件操作相关模块，用它可以查找符合自己目的文件，支持通配符操作，不用遍历整个目录判断每个文件是不是符合。
import glob
# pprint分行打印，对于数据结构比较复杂、数据长度较长的数据，适合采用pprint()打印方式
from pprint import pprint

import pandas as pd
import numpy as np

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

import spacy
from scipy.spatial.distance import jensenshannon

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
%matplotlib inline
# plt.style修改绘图风格，plt.style.use('ggplot')据说是模仿ggplot（R语言的一个的流行绘图包）的美学
# plt.style.available 是包含所有可用style的列表
plt.style.use("dark_background")

import joblib

from ipywidgets import interact, Layout, HBox, VBox, Box
import ipywidgets as widgets

from tqdm import tqdm
from os.path import isfile

import seaborn as sb

from lda import guidedlda as glda
from lda import glda_datasets as gldad
```

## 4. LDA做什么？（What does LDA do?）
在`LDA主题建模方法`中，`document（文档）`是`topics（主题）`按一定比例构成的集合，并且每个`topic`是`keywords（关键字）`按一定比例构成的集合。  
一旦您为算法提供了主题数量，它就会重新排列
- **文档中的主题分布(the topics distribution within the documents)**
- **主题内的关键字分布(keywords distribution within the topics)** 

以获得`topic-keywords分布`的良好组合。 

Q:当我说主题时，它实际上是什么？以及如何表示？  
A:一个主题只不过是典型代表的**主导关键词(dominant keywords)** 集合。
只需查看关键字，您就可以确定主题的内容。 

以下是获得良好**分离主题(segregation topics)** 的关键因素：
- 文本处理的质量。
- 文本谈论的各种主题。
- 主题建模算法的选择。
- 提供给算法的主题数量。
- 算法参数调整。

## 5.准备好停用词（Prepare Stopwords）
我们已经下载了停用词，让我们导入它们并使其在`stop_words`上可用。


```python
# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
```

## 6.导入数据（Import Data）


```python
df = pd.read_csv('donations.csv')
df.head()
```

## 7.DataFrame转list


```python
data = df.v1.values.tolist()
pprint(data[:1])
```

## 8.标记单词和清理文本（Tokenize words and Clean-up text）
让我们将每个句子标记为一个单词列表，删除标点符号和不必要的字符。


```python
# 文档分词
def sent_to_words(sentences):
    for sentence in sentences:
        # gensim.utils.simple_preprocess将文档转换为【由小写的词语组成的列表】,并忽略太短或过长的词语。
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data_words = list(sent_to_words(data))

pprint(data_words[:1])
```

## 9.创建Bigram和Trigram模型（Creating Bigram and Trigram Models）
- Bigrams是文档中经常出现的2个词
- Trigrams是经常出现3个单词。  

Gensim的`Phrases`模型可以构建和实现bigrams，trigrams，quadgrams等。  
`Phrases`两个重要的参数是`min_count`和`threshold [ˈθreʃhəʊld]`。这些参数的值越高，将单词组合成双字母组的难度就越大。


```python
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
pprint(trigram_mod[bigram_mod[data_words[0]]])
```

## 10.删除停用词，制作Bigrams和词形还原（Remove Stopwords, Make Bigrams and Lemmatize）
Bigram模型准备就绪。让我们定义函数来:
* 删除停用词
* 制作Bigrams和词形还原
* 并按顺序调用它们


```python
# 去除停用词
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

# 二元分词
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

# 三元分词
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# spacy用于进行分词，命名实体识别，词性识别
# -----------代替？---------
# import spacy
# nlp = spacy.load('en_core_web_sm')# 加载预训练模型
# --------------------------
# 安装问题待解决
import en_core_web_sm
nlp = en_core_web_sm.load()
# 对指定词性的词进行词形还原并添加到新列表
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
```


```python
# Remove Stop Words(得到对data_words去除停用词后的词表data_word_nostops)
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams(得到对data_word_nostops二元分词后的词表data_words_bigrams)
data_words_bigrams = make_bigrams(data_words_nostops)

# Do lemmatization keeping only noun, adj, vb, adv(得到对data_words_bigrams词性还原后的词表data_lemmatized)
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

pprint(data_lemmatized[:1])
```

## 11.创建主题建模所需的词典和语料库（Create the Dictionary and Corpus needed for Topic Modeling）
LDA主题模型的两个主要输入是 **字典（id2word）** 和 **语料库（corpus）** 。  ([Gensim官方教程翻译（二）——语料库与向量空间（Corpora and Vector Spaces）](https://blog.csdn.net/questionfish/article/details/46739207))  

让我们创造它们(借助`gensim.corpora.dictionary`方法，Dictionary类为每个出现在语料库中的单词分配了一个独一无二的整数编号）。  

- 字典（Dictionary）是由键值对token-id组成的字典（dict），可通过`dictionary.token2id`查看
- 语料库（Corpus）根据字典将文档（text）转换得到的稀疏向量表


```python
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized
# Term Document Frequency(生成稀疏文档向量语料库)
corpus = [id2word.doc2bow(text) for text in texts]

# View（打印第一篇文章的词频）
pprint(corpus[:1])
```


```python
id2word[0]
```


```python
# Human readable format of corpus (term-frequency)人性化显示第一篇文章的词频
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]
```

## 12.构建主题模型（ Building the Topic Model）
>在LDA主题建模方法中，document（文档）是topics（主题）按一定比例构成的集合，并且每个topic是keywords（关键字）按一定比例构成的集合。
一旦您为算法提供了主题数量，它就会重新排列**文档中的主题分布(the topics distribution within the documents)** 和**主题内的关键字分布(keywords distribution within the topics)**，以获得topic-keywords分布的良好组合。

我们拥有训练LDA模型所需的一切。除语料库和字典外，您还需要提供以下参数：

- 主题数量`num_topics`(topic-keywords数量）
- `alpha`和`eta`是影响主题稀疏性的超参数。根据Gensim文档，都默认 1.0 / num_topics之前。
- `chunksize`是每个训练块中使用的documents（文档）数
- `update_every`确定模型参数的更新频率
- `passes`是训练通过的总数（topic-keywords分布中每个topic下的keywords数量）。


```python
# Build LDA model
NUM_TOPICS = 5
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=NUM_TOPICS, 
                                           random_state=7,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

lda_model.save('model10.gensim')
```

## 13.查看LDA模型中的主题（View the topics in LDA model）

其中每个topic是keyword的组合，并且每个keyword对topic贡献一定的**权重(weightage)** 。

您可以使用`lda_model.print_topics()`看到每个topic的keyword以及每个keyword的权重 ，如下所示。


```python
# Print the Keyword in the every topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
```

权重反映了关键字对该主题的重要程度。

看看这些关键词，您能猜出这个主题是什么吗？您可以将其概括为“汽车”或“汽车”。  
同样，您是否可以浏览剩余的主题关键字并判断主题是什么？

![](https://www.machinelearningplus.com/wp-content/uploads/2018/03/Inferring-Topic-from-Keywords.png?ezimgfmt=ng:webp/ngcb3)

## 14.计算模型复杂度和相干性分数（Compute Model Perplexity and Coherence Score）
`模型复杂度(Model perplexity)`和`主题相干性(topic coherence)`提供了一种便捷的方法来判断给定主题模型的好坏程度。  
根据我的经验，尤其是[**主题相干性分数**](https://rare-technologies.com/what-is-topic-coherence/) 更为有用。


```python
# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, id2word=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
```

## 未知


```python
# os.getcwd()方法用于返回当前工作目录。
print(os.getcwd())
```


```python
mixture = [dict(lda_model[x]) for x in corpus]
pd.DataFrame(mixture).to_csv("topic_mixture.csv")
```


```python
top_words_per_topic = []
for t in range(lda_model.num_topics):
    top_words_per_topic.extend([(t, ) + x for x in lda_model.show_topic(t, topn = 20)])

pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("top_words.csv")
```


```python
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus, 
                                           num_topics = NUM_TOPICS, 
                                           id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
```


```python
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)
```


```python
# Show graph
'''    
Pick model (num topics) with the highest coherence score before flattening out.
'''
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
```


```python
# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
```


```python
# Select the model and print the topics
optimal_model = model_list[1]
model_topics = optimal_model.show_topics(formatted=False)
print(optimal_model.print_topics(num_words=10))
```

## 15.topics-keywords可视化（Visualize the topics-keywords）
现在已经建好了LDA模型，下一步是检验生成的topics和与之关联的keywords。  
没有什么工具比`pyLDAvis`包的交互式图表更好用了，并且它可以很好的与jupyter notebook一起工作。

#### PS：那么如何推断pyLDAvis的输出呢？
![](https://www.machinelearningplus.com/wp-content/uploads/2018/03/pyLDAvis.png?ezimgfmt=ng:webp/ngcb3)


上图中的每个气泡(bubble)代表一个主题。气泡越大，该主题就越流行(prevalent)。
一个好的主题模型将在整个图表中分散相当大（范围）的、非重叠的气泡，而不是聚集(clustered)在一个象限(quadrant)中。  

具有太多主题的模型通常会有许多重叠，小尺寸的气泡聚集在图表的一个区域中。
如果将光标移动到其中一个气泡上，右侧的单词和条形将会更新。这些单词是构成所选主题的显著关键字(the salient keywords)。

我们已经成功构建了一个好的主题模型。鉴于我们之前对文档中自然主题数量的了解，找到最佳模型非常简单。


```python
# Visualize the topics
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis
```

## 16.构建LDA Mallet模型（Building LDA Mallet Model）
到目前为止，您已经看到了Gensim内置的LDA算法版本。然而，Mallet的版本通常会提供更高质量的topic。

Gensim提供了一个包装器，用于在Gensim内部实现Mallet的LDA。  
>You only need to download the zipfile, unzip it and provide the path to mallet in the unzipped directory to gensim.models.wrappers.LdaMallet.

您只需要：
1. 下载 [ZIP](https://www.machinelearningplus.com/wp-content/uploads/2018/03/mallet-2.0.8.zip) 文件
2. 将其解压缩
3. 把它的路径提供给在解压缩目录下的mallet
4. 把mallet路径提供给`gensim.models.wrappers.LdaMallet`

看看我在下面如何做到这一点。

PS:只需改变LDA算法，我们就可以将相干性得分（coherence score）从.53增加到.63。不错！


```python
import os
from gensim.models.wrappers import LdaMallet

# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
os.environ.update({'MALLET_HOME':r'C:/new_mallet/mallet-2.0.8/'})
mallet_path = 'C:/new_mallet/mallet-2.0.8/bin/mallet' # update this path

ldamallet = LdaMallet(mallet_path, corpus=corpus, num_topics=8, id2word=id2word)
```


```python
# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
pprint('\nCoherence Score: ', coherence_ldamallet)
```

## 17.如何找到LDA的最佳主题数量？（How to find the optimal number of topics for LDA?）
我寻找最佳主题数的方法是：构建具有不同主题数量（k）的许多LDA模型，并选择具有最高相干性值(the highest coherence value)的LDA模型。

- 选择一个标志着快速增长的主题相干性的峰值“k”，通常会提供有意义和可解释的主题。  
- 选择更高的值有时可以提供更细粒度(granular)的子主题。
- 如果您看到相同的关键字在很多个主题中重复出现，则可能表示'k'太大。

用`compute_coherence_values()`（见下文）训练多个LDA模型，并提供模型及其对应的相干性分数(coherence scores)。


```python
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
```


```python
# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)
```


```python
# Show graph
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
# demo:https://www.machinelearningplus.com/wp-content/uploads/2018/03/Choosing-the-optimal-number-of-LDA-topics-2.png?ezimgfmt=ng:webp/ngcb3
```


```python
# 选择最佳数量的LDA主题
# Print the coherence scores
for m, cv in zip(x, coherence_values):
    pprint("Num Topics =", m, " has Coherence Value of", round(cv, 4))
```


```python
# 如果相关性得分似乎在不断增加，那么选择在展平之前给出最高CV的模型可能更有意义。这就是这种情况。
# 因此，对于进一步的步骤，我将选择具有20个主题的模型。

# Select the model and print the topics
optimal_model = model_list[1]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))
```


```python
# Wordcloud of Top N words in each topic
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)
topics = optimal_model.show_topics(formatted=False)
fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)
for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')
plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()
```

## 18.在每个句子中找到主要话题（Finding the dominant topic in each sentence）
主题建模的一个实际应用是确定给定文档的主题。

为了找到这个，我们找到该文档中贡献百分比最高的主题编号。

下面的函数很好地将此信息聚合在一个可呈现的表中。`format_topics_sentences()`


```python
def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(10)
```

## 19.找到每个主题最具代表性的文档（Find the most representative document for each topic）
有时，主题关键字(topic keywords)可能不足以理解主题的含义。因此，为了帮助理解该主题，您可以找到给定主题最有代表性的的文档(document)，并通过阅读该文档来推断该主题。


```python
# Group top 5 sentences under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet.head()
```

## 20.将主题分发给各文档（Topic distribution across documents）
最后，我们希望了解主题的数量和分布，以判断讨论的范围。


```python
# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = sent_topics_sorteddf_mallet[['Topic_Num', 'Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics
```


```python
# dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
# corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model10.gensim')
# lda10 = gensim.models.ldamodel.LdaModel.load('model10.gensim')
# lda_display10 = pyLDAvis.gensim.prepare(lda10, corpus, dictionary, sort_topics=False)
# pyLDAvis.display(lda_display10)
```


```python
# import pyLDAvis.gensim
#visual graph
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
# # lda matching by words vs matching by documents. Document may contains multiple topics and words. 
pyLDAvis.display(lda_display)
```


```python
seed_topic_list = [['covid', 'team', 'win',  'victory'],
                   ['percent', 'company', 'market', 'price', 'sell', 'business', 'stock', 'share'],
                   ['donation', 'hunger', 'cause', 'world'],
                   ['political', 'government', 'leader', 'official', 'state', 'country',
                    'american','case', 'law', 'police', 'charge', 'officer', 'kill', 'arrest', 'lawyer']]

```


```python
# '''Make our own dataset and word2id'''

# # print(X.shape)
# # print(corpus[100])
word2id = {}
vocab = []
index = 0
for tx in texts:
    for word in tx:
        if word not in word2id:
            vocab.append(word)
            word2id[word] = index
            index += 1

print(len(word2id))


```


```python
# ## transfer corpus to word_ids sentences
corpus_with_id = []
max_len = max([len(x) for x in corpus])
for line in corpus:
    doc = []
    for word, fre in line:
        doc.append(word)
    doc += [0 for _ in range(max_len - len(doc))]
    corpus_with_id.append(doc)

import numpy
corpus_with_id = numpy.array(corpus_with_id)
print(corpus_with_id.shape)

```


```python
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        print(word)
```


```python
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        if word in word2id:
            seed_topics[word2id[word]] = t_id
```


```python
# '''model training'''
model = glda.GuidedLDA(n_topics=5, n_iter=100, random_state=7, refresh=20)
model.fit(corpus_with_id, seed_topics=seed_topics, seed_confidence=0.15)
```


```python
# '''Get guidedLDA output'''
n_top_words = 10
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
```


```python
# '''Retreive the document-topic distributions'''
doc_topic = model.transform(corpus_with_id)
for i in range(7):
    print("top topic: {} Document: {}".format(doc_topic[i].argmax(),
                                                  ', '.join(np.array(vocab)[list(reversed(corpus_with_id[i,:].argsort()))[0:5]])))
```


```python
model.purge_extra_matrices()
```

#### pickle模块
python的pickle模块实现了python的所有数据序列和反序列化。基本上功能使用和JSON模块没有太大区别，方法也同样是dumps/dump和loads/load。cPickle是pickle模块的C语言编译版本相对速度更快。

与JSON不同的是pickle不是用于多种语言间的数据传输，它仅作为python对象的持久化或者python程序间进行互相传输对象的方法，因此它支持了python所有的数据类型。

pickle反序列化后的对象与原对象是等值的副本对象，类似与deepcopy。

参考文章：[python 序列化之JSON和pickle详解](https://www.cnblogs.com/tkqasn/p/6005025.html)


```python
from six.moves import cPickle as pickle
with open('guidedlda_model.pickle', 'wb') as file_handle:
     pickle.dump(model, file_handle)
# load the model for prediction
with open('guidedlda_model.pickle', 'rb') as file_handle:
     model = pickle.load(file_handle)
doc_topic = model.transform(corpus_with_id)

```


```python
numpy.savetxt("guidedlda.csv", doc_topic, delimiter=",")
```

## 21.总结（Conclusion）
从了解主题建模可以做什么入手，我们使用Gensim的LDA构建了一个基本主题模型，并用pyLDAvis使主题可视化。

然后我们构建了基于mallet的LDA实现。您了解了如何使用相干性分数(coherence scores)找到最佳主题数量，以及如何理解如何选择最佳模型。

最后，我们看到了如何整合(aggregate)和呈现(present)结果，以产生可能更具可操作性(actionable)的见解。
