# 0.思维导图
[XMIND下载](https://file.zhihuishu.com/zhs/ablecommons/demo/202010/19e7d1a4b9934ac5b2d39eed0f0f4c51.xmind "XMIND下载")
[PNG下载](https://image.zhihuishu.com/zhs/ablecommons/demo/202010/ca49d81f8a4e4e89b1d18f2f7e3b4787.png "PNG")
# 1.背景
近年来，很多用户都喜欢使用在线新闻网站和手机 APP 来进行电子化的新闻阅读。但是，由于每天都有大量新闻产生和发布，用户难以在有限的时间内从大量新闻中找到自己感兴趣的内容，面临严重的新闻信息过载。
个性化新闻推荐可以根据用户的个人兴趣对候选新闻进行排序和展示，是提升用户在线新闻阅读体验的一项重要技术。目前，新闻推荐已广泛用于Google News、Yahoo! News、Microsoft News、新浪、网易、今日头条等诸多在线新闻平台中。
近年来，学术界和工业界的研究人员提出一些基于深度学习的新闻推荐方法，如 Embedding based News Recommendation 等。但**这些方法通常是在私有数据集上设计和验证的，这使得其他研究人员难以对这些方法进行验证并提出改进。**
很多推荐任务如产品推荐、电影推荐和餐厅推荐等通常有一些公认的基准数据集，例如 Amazon、MovieLens、Yelp 等。然而**在新闻推荐领域高质量的基准数据集比较匮乏，严重制约了这一研究领域的进展。**
![](https://www.msra.cn/wp-content/uploads/2020/07/mind-1.png)
### 新闻推荐同其他推荐系统的区别
新闻文章的推荐与音乐和电影等其他知名类型的推荐器系统在几个方面有所不同。主要区别在于：
（i）有时新鲜度比相关性更重要；
（ii）新闻文章之间的相似性并不一定意味着它们是相关的——尽管新闻文章可能共享许多单词，但它们之间并不相关；
（iii）新闻故事的非结构化格式与具有结构化属性的其他对象（例如，社交网络）相比，分析起来更困难；
（iv）新闻阅读者可能对新闻文章中包含的某些特定事件有特别的偏好；
（v）偶然性（即，推荐新闻文章中的变化）；
（vi）突破性和潮流，即使新闻文章与用户的总体兴趣无关，也可能会引起他们的兴趣。
### 面临的挑战
尽管推荐系统总体上取得了长足进步，但是仍然存在挑战，这些挑战限制了当前可用解决方案在针对移动用户的新闻推荐方面的有效性 。例如，在新闻推荐领域，以下挑战 是重大的：
（a）移动设备屏幕的用户界面可用空间有限；
（b）新闻文章的生命周期短；
（c）当冷启动用户第一次请求推荐时的挑战是，他们的兴趣最初是未知的；
（d）新闻冷启动的挑战，这是指难以推荐与许多用户的偏好无关的新文章的困难；
（e）关于用户真正喜欢哪篇文章的明确信号很难找到，无法轻松的检测到用户对新闻文章的需求；
（f）尚未就如何评估新闻推荐系统建立共识；
（g）用户对新闻推荐系统的偏好特定文章不仅取决于主题和命题内容，还取决于用户当前的上下文。用户的当前上下文可以涵盖各种类型的信息，例如用户的当前位置，访问时间，社交环境和外部事件。

# 2.英文数据集

### UCI知识库-Anonymous Microsoft Web Data（非新闻数据）
**简介：**UCI知识库(http://archive.ics.uci.edu/ml/) 是Blake等人在1998年开放的一个用于机器学习和评测的数据库。目前包含3个可用于新闻推荐的数据集:
- [Anonymous Microsoft Web Data](http://archive.ics.uci.edu/ml/datasets/Anonymous+Microsoft+Web+Data "Microsoft")数据集，是38,000个用户在1998年2月某周内在微软主页的全部阅读历史记录；
- [MSNBC](http://archive.ics.uci.edu/ml/datasets/MSNBC.com+Anonymous+Web+Data "MSNBC")数据集，是989,818个用户在1999年9月28日在msnbc.com上的阅读历史记录；
- [Syskill&Webert](http://archive.ics.uci.edu/ml/datasets/Syskill+and+Webert+Web+Page+Ratings "Syskill&Webert")数据集，记录的是单个用户的阅读及评分历史记录。

### Digg数据集（缺乏新闻内容）
**简介：**Digg数据集，是由美国南加州大学信息科学研究所收集的。收集的新闻是2009年6月份Digg网站首页的3,553条新闻。表`digg_votes`中包含139, 409个用户对3,553条新闻的累计3,018,197个投票，表`digg_friends`包含71,367个用户之间的1,731 ,658个链接关系。
**数据源：**一家美国社交新闻网站，也是第一个掘客类网站。[Digg - What the Internet is talking about right now](http://digg.com "Digg - What the Internet is talking about right now")
**发布年月：**2009年
**是否公开：**是（下载地址：https://www.isi.edu/~lerman/downloads/digg2009.html 解压密码：digg2009_user）
**数据集格式：**
- `digg_votes.csv`

| Column | Content |
| ------------ | ------------ |
| vote_date  | 投票时间戳 |
| voter_id | 投票者的匿名唯一id |
| story_id | 故事的匿名唯一id |

- `digg_friends.csv`

| Column | Content |
| ------------ | ------------ |
| mutual | 表明链接是否具有共同的朋友关系 (1) or not (0) |
| friend_date | 创建友情链接时的时间戳 |
| user_id | 用户的匿名唯一id |
| friend_id | 朋友的匿名唯一id |

**存在问题：**该数据集删除了新闻标题和正文，且story_id并非新闻真实id，无法得知新闻具体内容。

### Yahoo!
**简介：**它包含14180篇新闻文章和34022次点击事件。每篇新闻文章都由单词 id 表示，不提供原始新闻文本。此数据集中的用户数量未知，因为没有用户ID。
**数据源：**http://news.yahoo.com
**发布年月：**2016年
**是否公开：**是（下载地址：[L22 - Yahoo! News Sessions Content, version 1.0 (16 MB)](https://webscope.sandbox.yahoo.com/catalog.php?datatype=l "L22 - Yahoo! News Sessions Content, version 1.0 (16 MB)")）
**数据划分：**未知
**数据集格式：**未知
**参考文章：**
[雅虎新闻的经典新闻推荐系统论文解析](https://zhuanlan.zhihu.com/p/115856168 "雅虎新闻的经典新闻推荐系统论文解析")
### MIND(MIcrosoft News Dataset)
**简介：**MIND 数据集是微软公司从六周内（October 12 - November 22, 2019） Microsoft News 用户的匿名化新闻点击记录中构建的，它包含16万多条新闻条目，1500 余万次展示记录，以及来自100万匿名用户的2400余万次点击行为。
**数据源：**[Microsoft News](https://microsoftnews.msn.com/ "Microsoft News")
**发布年月：**2020年7月
**是否公开：**是（下载地址：https://msnews.github.io/ ）
**数据划分：**
- 训练集：第五个星期的数据(2,186,683个Samples)。
- 测试集：最后一个星期的数据(2,341,619个Samples)
- 验证集：取第五个星期的最后一天做验证集(365,200个Samples)
- ClickHist：在训练集中选择了前四个星期的行为去构造，在测试集中则选择五个星期的行为去构造Click History。且在划分数据的过程中把ClickHist列表为空的剔除。

**统计信息：**

| Item | Scale |
| ------------ | ------------ |
| Users |  1,000,000 |
| News | 161,013 |
| News category | 20 |
| Impression | 15,777,377 |
| Click behavior | 24,155,470 |
| size | 1.15G |
| Entity | 3,299,687 |
| Avg.title len| 11.52 |
| Avg.abstract len| 43.00 |
| Avg.body len| 585.05 |

**数据集格式：**
每个数据集都是一个压缩包文件夹，包括四个文件

File Name | Description
------------- | -------------
behaviors.tsv  | The click histories and impression logs of users
news.tsv  | The information of news articles
entity_embedding.vec    | The embeddings of entities in news extracted from knowledge graph
relation_embedding.vec    | The embeddings of relations between entities extracted from knowledge graph

#### 【behaviors.tsv】

该文件中每个样本点有5个数据项，用制表符分割。

* Impression ID.
* User ID.
* Time. 格式为"月/日/年 时:分:秒 AM/PM".
* History. 该用户在这条impression之前的新闻点击历史(ID list of clicked news)，点击过的新闻按时间排序.
* Impressions. 该impression下显示的新闻列表 及 用户对其的点击行为 (1代表点击，0代表未点击)， impressions中的新闻顺序是打乱的.

Column | Content
------------- | -------------
Impression ID | 91
User ID | U397059
Time | 11/15/2019 10:22:32 AM
History | N106403 N71977 N97080 N102132 N97212 N121652
Impressions | N129416-0 N26703-1 N120089-1 N53018-0 N89764-0 N91737-0 N29160-0

#### 【news.tsv】
该文件包含`behaviors.tsv`文件中涉及的新闻文章的详细信息。它有7个数据项，用制表符分割。

* News ID（新闻ID）
* Category（类别）
* SubCategory（子类别）
* Title（标题）
* Abstract（摘要）
* URL（文章链接）
* Title Entities（标题中包含的实体）
* Abstract Entities（摘要中包含的实体）

由于许可结构的原因，无法下载MSN新闻文章的完整内容。但是，为方便起见，我们提供了一个[utility script](https://github.com/msnews/MIND/tree/master/crawler) 来帮助从数据集中的MSN URL解析新闻网页。 但由于时间限制，某些URL已过期且无法成功访问。

Column | Content
------------- | -------------
News ID | N37378
Category | sports
SubCategory | golf
Title | PGA Tour winners
Abstract | A gallery of recent winners on the PGA Tour.
URL | https://www.msn.com/en-us/sports/golf/pga-tour-winners/ss-AAjnQjj?ocid=chopendata
Title Entities | [{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": [0], "SurfaceForms": ["PGA Tour"]}]
Abstract Entites | [{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": [35], "SurfaceForms": ["PGA Tour"]}]

下表列出了“实体”（Entities）列中字典键的描述:

Keys | Description
------------- | -------------
Label | The entity name in the Wikidata knwoledge graph
Type | The type of this entity in Wikidata
WikidataId | The entity ID in Wikidata
Confidence | The confidence of entity linking
OccurrenceOffsets | The character-level entity offset in the text of title or abstract
SurfaceForms | The raw entity names in the original text

#### 【entity_embedding.vec】 & 【relation_embedding.vec】
这两个文件包含通过TransE方法从子图（从[WikiData知识图谱](https://www.wikidata.org/wiki/Wikidata:Main_Page "WikiData知识图谱")）获知的实体和关系的100维嵌入。在两个文件中，第一列是实体/关系的ID，其他列是嵌入矢量值。我们希望这些数据可以促进对knowledge-aware news recommendation的研究。示例如下所示：

ID | Embedding Values
------------- | -------------
Q42306013 | 0.014516	-0.106958	0.024590	...	-0.080382

由于从子图中学习embedding的某些原因，entity_embedding.vec文件中少部分实体可能不具有embeddings。

**（附1）小版本：MIND-small**
随机抽样50,000个用户及其行为日志，仅包含训练集和验证集。
**（附2）MIND数据集与其他新闻推荐数据集对比：**
![](https://img-blog.csdnimg.cn/20200724233012344.png)
- 相比于其他的数据集，MIND的数据量是非常庞大的，这对于模型的训练提升帮助是非常大的。由Microsoft发表的论文表明大数据集对算法的研究改进有极大的帮助。
- 可以从表中看出，MIND数据集使用的是英文，显然相较于其他语言的数据集能够更好地契合当下新闻推荐系统。
- MIND数据集的News information也非常的丰富，这样能够使模型学到更加准确的文本表示。

**（附3）MIND上一些算法的比较**
![](https://upload-images.jianshu.io/upload_images/16043538-935415b73a88f386.png?imageMogr2/auto-orient/strip|imageView2/2/w/724/format/webp)

**参考文章**
[ACL 2020 | 微软发布大规模新闻推荐数据集MIND，开启新闻推荐比赛](https://www.msra.cn/zh-cn/news/features/acl-2020-mind "ACL 2020 | 微软发布大规模新闻推荐数据集MIND，开启新闻推荐比赛")
[MIND- A Large-scale Dataset for News Recommendation.pdf](https://file.zhihuishu.com/zhs/ablecommons/demo/202010/fdf33f7d504a4f089b9e555badbbbc45.pdf "MIND- A Large-scale Dataset for News Recommendation.pdf")

# 3.中文数据集
### 财新网
**简介：**2014年中国计算机学会主办的“第二届中国大数据技术创新大赛”中公开了由财新网提供的数据集。包括10,000个用户在2014年3月的所有新闻浏览记录。
**数据源：**财新网
**发布年月：**2014年
**是否公开：**是（官方已关闭下载通道）
**数据划分：**以3月20号为界限，前20天的数据(83209条)作为训练数据， 后10天的数据(18995条)作为测试数据。
**统计信息：**

| Item | Scale |
| ------------ | ------------ |
| 浏览记录 | 116,224条 |
| 数据集大小 | 201M |
| 用户数 | 10,000 |
| 出现新闻数 | 6183条 |

**数据集格式：**
共有五个域：用户编号、新闻编号、访问页面的时间(Unix时间戳)、新闻标题、新闻正文。

|user_id|news_id|read_time|news_title|news_content|
|-|-|-|-|
|5218791|100648598|1394463264|消失前的马航370|【财新网】（实习记者葛菁）据新华社消息，马来西亚航空公司表示...|
|5218791|100648802|1394463205|马航代表与乘客家属见面|3月9日，马来西亚航空公司代表在北京与马航客机失联事件的乘客家属见面。|

**参考文章：**
[https://github.com/buptweixin/RecommendSystem](https://github.com/buptweixin/RecommendSystem)
![](https://image.zhihuishu.com/zhs/ablecommons/demo/202010/9e59c048084c40108f96b5ba337200db.png)
# 4.其他语言
### Plista(德语)
**简介：**在ACM RecSys [2013新闻推荐系统研讨会和挑战赛（International News Recommender Systems Workshop and Challenge）](http://recsys.acm.org/recsys13/nrs/) 的背景下 ，我们于2013年7月1日发布了一个数据集，其中包含2013年6月份14,897,978个用户的84,210,795条交互记录，70,353篇新闻文章和1,095,323次点击。
**数据源：**通过收集13个德国新闻门户网站上发表的新闻文章和用户的点击日志，构建了 Plista4数据集，该数据集中的新闻文章为德语，用户主要来自德语国家。
**发布年月：**2013年7月
**是否公开：**是（~~下载地址：https://sites.google.com/site/newsrec2013/challenge~~ 官网似乎已无法下载）
### Adressa(挪威语)
**简介：**
Adressa数据集是一个新闻数据集，其中包括与匿名用户有关的新闻报道（挪威语）。我们希望该数据集将有助于与读者一起更好地理解新闻报道。
该数据集是与挪威科技大学（NTNU）和Adressavisen（位于挪威特隆赫姆的当地报纸）合作发布的，这是RecTech关于推荐技术项目的一部分。
> 整体来讲，Adressa是内容最为全面的，可以做常规的新闻推荐，也可也基于session做，也可以探究基于知识图谱的推荐。

**数据源：**http://www.adresseavisen.no
**发布年月：**2017年？
**是否公开：**是（下载地址： [SmartMedia Adressa dataset](http://reclab.idi.ntnu.no/dataset/ "SmartMedia Adressa dataset") ）
**数据集格式：**
整个数据集分为规模不同的两个版本，1周流量1.4GB数据集，以及10周流量16GB数据集。
- 1.4GB数据集包含11,207篇新闻文章、561,733个用户和2,286,835个点击事件;
- 10GB数据集包含48,486篇新闻文章、3,083,438个用户和27,223,576个点击事件。

~~每个点击事件包含几个属性，如会话时间、新闻标题、新闻类别和用户 ID。每篇新闻文章都与作者、实体和主体等详细信息相关联。~~
轻型版本`one_week.tar.gz`的数据集保存在一个文件夹中并只包含基本属性。
完整版本`three_month.tar.gz`的数据集包含3个文件夹：

| FolderName | Content |
| :------------ | :------------ |
| rawdata | 原始数据 |
| artdata | 文章相关数据 |
| contentdata | 文章内容信息 |

详见：[Addressa Dataset Description.pdf](https://file.zhihuishu.com/zhs/ablecommons/demo/202010/0e8e017b91be4c8696372a2b8ab5bf41.pdf "Addressa Dataset Description.pdf")
**参考文章：**
[The Adressa dataset for news recommendation](https://dl.acm.org/doi/10.1145/3106426.3109436 "The Adressa dataset for news recommendation")
[《The Adressa Dataset for News Recommendation》的理解](https://zhuanlan.zhihu.com/p/73700480 "《The Adressa Dataset for News Recommendation》的理解")
[Graph Neural News Recommendation with Unsupervised Preference Disentanglement.pdf](https://file.zhihuishu.com/zhs/ablecommons/demo/202010/0d0680295b6540ee8c40dfa92044b032.pdf "Graph Neural News Recommendation with Unsupervised Preference Disentanglement.pdf")
**（附1）如果您使用该数据集，请引用以下文章：**
> Gulla, J. A., Zhang, L., Liu, P., Özgöbek, Ö., & Su, X. (2017, August). The Adressa dataset for news recommendation. In Proceedings of the International Conference on Web Intelligence (pp. 1042-1048). ACM.

### Globo(葡萄牙语)
**简介：**这个数据集包含大约314,000个用户，46,000篇新闻文章和300万次点击记录。每个单击记录都包含用户 ID、新闻 ID 和会话时间等字段。最早开放在Kaggle平台上，提供训练好的新闻embedding，没有原始的新闻文章信息。
**数据源：**巴西流行新闻门户网站 [globo](https://www.globo.com/ "globo")
**发布年月：**2018年
**是否公开：**是（下载地址：[News Portal User Interactions by Globo.com](https://www.kaggle.com/gspmoreira/news-portal-user-interactions-by-globocom "News Portal User Interactions by Globo.com")）

# 5.总结及思考
- 新闻推荐数据集基本构成：编号，用户ID，操作时间，新闻ID（标题，时间，摘要/内容），历史记录
- 以上数据集均为由内部公开的用户行为日志，似乎无法从外部爬取数据集？
- 能否用博文代替新闻，用用户转发代替用户点击，以此训练模型？（参考文章：[基于社交网络的新闻推荐方法研究.pdf](https://file.zhihuishu.com/zhs/ablecommons/demo/202010/c00d606029564e868ad3adb7d4a4e5d0.pdf)）

# 6.新闻推荐相关会议
[International Workshop on News Recommendation and Analytics(INRA)](https://www.ntnu.no/wiki/pages/viewpage.action?pageId=188580623#app-switcher "International Workshop on News Recommendation and Analytics(INRA)")