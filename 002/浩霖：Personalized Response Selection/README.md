# Personalized Response Selection
根本目的：为了使对话机器人更具有个人特色（个性），或者指定一种角色，使聊天机器人变得有趣进而对用户更有吸引力。

### 1.Task Definition
---------------------

**在基于检索式的聊天模型中，通过指定的persona与上文的对话，在给定的候选答复中选出最合适的答复，要求要尽量满足聊天的角色一致性。**

实现个性化回复的方法一般有生成式与检索式，Personalized Response Selection则主要是在检索式聊天模型中进行构建一种分类模型。

### 2.Datasets PERSONA-CHAT

PERSONA-CHAT数据集是为了个性化对话任务而构建的数据集，其主要由personas 与 dialogues组成

**2.1 Personas**

首先确定一些角色personas,这些personas分别由一些短句子描述，如下表左侧的两个original persona，为了避免对话中某些词的重叠（这可能使任何后续的机器学习任务不那么具有挑战性，并且解决方案不会推广到更复杂的任务）后续增加了revised personas

![](https://image.zhihuishu.com/zhs/ablecommons/demo/202010/966b6ef13c1c4253a5c51bc298d70422.jpg)

**2.2 Dialogues**

给采集者分配不同的Persona,两两配对进行对话，要求他们的聊天内容必须反映出其分别所扮演的角色特点且对话不能与所给定的描述persona的sentences有明显的相似度如下表

![](https://image.zhihuishu.com/zhs/ablecommons/demo/202010/971818ea2c194ab08bb2a04449d6c2da.jpg)

**2.3 Summary**

最终我们得到每个persona的描述性语句集与在此基础上生成的多轮对话数据集。