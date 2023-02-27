# -*- coding:utf-8 -*-
import datetime

import jieba
import re
import pprint
from gensim import corpora
from collections import defaultdict
from gensim import models
from gensim import similarities

data_lists = []

# json文件
# filename2 = 'data10.json'
# with open(filename2, 'r', encoding='utf8') as f:
#     m = json.load(f)  # json.load() 这种方法是解析一个文件中的数据
#     for i in m:
#         text = re.sub(r"[^\u4e00-\u9fa5 ]+", '', i['fullText'])
#         data_list = " ".join(jieba.lcut(text))  # 精简模式，返回一个列表类型的结果
#         data_lists.append(data_list)
#     text_corpus = data_lists
#     # print(text_corpus)
print(datetime.datetime.now())
# 非标准的json文件
filename2 = 'data_20k.json'
# time_start = datetime.now()
with open(filename2, 'r', encoding='utf8') as f:
    m = f.readlines()
    for i in m:
        c = eval(i)
        text = re.sub(r"[^\u4e00-\u9fa5 ]+", '', c['fullText'])
        data_list = " ".join(jieba.lcut(text))  # 精简模式，返回一个列表类型的结果
        data_lists.append(data_list)
    text_corpus = data_lists
    # print(text_corpus)
# time_end = datetime.now()
# time_c = time_end - time_start  # 运行所花时间
# print('time cost', time_c, 's')

# 停用词
# 读取停顿词列表
stopword_list = [k.strip() for k in open('stopwords', encoding='utf8').readlines() if k.strip() != '']

# print(stopword_list)
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stopword_list]
         for document in text_corpus]
print(datetime.datetime.now())
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once 文档预处理，仅保留多次出现的字词
processed_corpus = [[token for token in text if frequency[token] > 10] for text in texts]
# pprint.pprint(processed_corpus)

dictionary = corpora.Dictionary(processed_corpus)
# Dictionary<786 unique tokens: ['一年', '一般', '七年', '三合', '上述事实']...>
# print(dictionary)
# 显示词频
# pprint.pprint(dictionary.token2id)

# 假设我们要矢量化短语“人机” 互动“。我们可以 使用字典的方法为文档创建词袋表示形式，该方法返回单词的稀疏表示形式 计数：doc2bow
# new_doc = "一 一万元"
# new_vec = dictionary.doc2bow(new_doc.lower().split())
# print(new_vec)
# [(193, 1), (689, 1)]

# 我们可以将整个原始语料库转换为向量列表：
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
# pprint.pprint(bow_corpus)

# # train the model 训练模型
# 使用模型对语料库进行转换。我们使用模型作为一个抽象术语，指的是从一个文档表示到另一个文档表示形式。
# 模型的一个简单示例是 tf-idf。tf-idf 模型 将向量从词袋表示转换为向量空间 其中频率计数根据 语料库中的每个单词。
tfidf = models.TfidfModel(bow_corpus)
# words = "一 一万元".lower().split()
# 第一个条目是 令牌 ID 和第二个条目是 tf-idf 权重。
# print(tfidf[dictionary.doc2bow(words)])

print(datetime.datetime.now())
# 要通过 TfIdf 转换整个语料库并对其进行索引，做相似性查询的准备：
# 并针对语料库中的每个文档查询我们的查询文档的相似性：query_document
index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=70000)
test_string = '张承兵减刑刑事裁定书重庆市第二中级人民法院刑事裁定书（2017）渝02刑更112号罪犯张承兵，男，汉族，生于1970年1月17日，重庆市梁平县人，现在重庆市三合监狱服刑。罪犯张承兵曾因犯重大责任事故罪于2004年6月10日被北京市门头沟区人民法院判处有期徒刑二年六个月，2006年1月20日刑满释放。2012年7月4日，北京市朝阳区人民法院作出（2012）朝刑初字第1430号刑事判决，以被告人张承兵犯贩卖毒品罪，判处有期徒刑十年，剥夺政治权利二年，并处罚金10000元（已缴纳）。判决发生法律效力后，于2013年1月9日交付执行。该犯于2015年6月5日经本院裁定减去有期徒刑一年的刑罚执行。现执行机关重庆市三合监狱于2017年2月27日以该犯在服刑期间确有悔改表现为由，提出减刑建议书，报送本院审核。本院依法组成合议庭进行了审理，现已审理终结。执行机关认为，罪犯张承兵在服刑中，确有悔改表现。其事实有监区鉴定、奖励审批表和罪犯小组评议等证据为证，建议予以减刑。经审理查明，罪犯张承兵在服刑中，能认罪悔罪；认真遵守法律法规及监规，接受教育改造；积极参加思想、文化、职业技术教育；积极参加劳动，努力完成劳动任务，现已获监狱表扬四次，确有悔改表现。上述事实，有罪犯张承兵的陈述、罪犯评审鉴定表、罪犯奖励审批表、改造鉴定、证人证言以及非税收入一般缴款书等证据予以证实。本院予以确认。本院认为，罪犯张承兵在服刑期间，确有悔改表现。该犯贩卖毒品数量较大，具有较大的社会危害性，综合考虑其犯罪性质、情节以及原判刑罚等情况，依照《中华人民共和国刑法》第七十八条、第七十九条，《最高人民法院关于办理减刑、假释案件具体应用法律的规定》第二条、第三条以及第六条之规定，裁定如下：对罪犯张承兵减去有期徒刑八个月的刑罚执行，剥夺政治权利二年不变（本次减刑后刑期至2020年4月16日止）。本裁定送达后即发生法律效力。审判长邹绍云代理审判员赵自辉代理审判员江平二〇一七年三月十三日书记员张惠中'
test_doc_list = [word for word in jieba.cut(test_string)]
test_doc_vec = dictionary.doc2bow(test_doc_list)
# print(test_doc_vec)
sims = index[tfidf[test_doc_vec]]
# print(sims)

# 换个输出格式
for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    print(document_number, score)
