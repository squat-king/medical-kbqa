# medical-kbqa
Medical question answering system based on knowledge mapping and NLP
基于知识图谱和NLP的智能问答系统
# Author: jhz(Kevin)
# Email：jhzsdufe@163.com
# Date: 2021-4-01
本文从医疗领域的角度出发，以疾病、食品、药品、生产制造等领域的专业知识为基础，结合用户关于对于医疗问题问句的句法特征，分析设计出了一款即时的问答系统。本文主要工作如下：
1、数据的获取与处理。以垂直网站（寻医问药网）为数据来源，通过爬虫获取到13000条网页数据，从中筛选并提取出关键信息8000余条，以半结构化数据的形式备份在数据库中。
2、知识图谱的构建。对医疗领域的半结构化数据进行分析和建模，提取出8种实体类型和11种实体关系。构建起以疾病为中心的医疗知识图谱，实体规模3万，实体关系规模44万。
3、命名实体识别算法模型的构建。使用结合双向长短期记忆网络(Bidirectional Long Short-Term Memory，BiLSTM)和条件随机场(Conditional Random Field，CRF) 的命名实体识别(Named Entity Recognition，NER)算法对问句进行命名实体识别，获得一个包含字符标签信息的序列，提取出用户问句中的关键信息。
4、句子相似度模型的构建。使用基于双向变压器编码器（Bidirectional Encoder Representations from Transformers,Bert）模型的相似度计算方法，计算用户问句和实体关系的关联，以获取最优回答。
5、设计并实现了基于知识图谱的问答系统。在理论和实验基础上，将知识图谱和基于深度学习的命名实体识别模型、句子相似度计算模型应用于问答系统，设计出交互模块，获取用户问题，并将答案结果以控制台输出的形式进行展示。

