import json
import jieba
import re
import pprint
from gensim import corpora
from collections import defaultdict
from gensim import models
from gensim import similarities

data_lists = []

filename2 = 'data10.json'
with open(filename2, 'r', encoding='utf8') as f:
    m = json.load(f)  # json.load() 这种方法是解析一个文件中的数据
    for i in m:
        text = re.sub(r"[^\u4e00-\u9fa5 ]+", '', i['fullText'])
        data_list = " ".join(jieba.lcut(text))  # 精简模式，返回一个列表类型的结果
        data_lists.append(data_list)
    text_corpus = data_lists
    # print(text_corpus)

# Create a set of frequent words 排除词
stoplist = set('的 一 中华人民共和国 人民币 他'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_corpus]

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

# 要通过 TfIdf 转换整个语料库并对其进行索引，做相似性查询的准备：
# 并针对语料库中的每个文档查询我们的查询文档的相似性：query_document
index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], num_features=700)
test_string = '张承兵减刑刑事裁定书重庆市第二中级人民法院刑事裁定书'
test_doc_list = [word for word in jieba.cut(test_string)]
test_doc_vec = dictionary.doc2bow(test_doc_list)
# print(test_doc_vec)
sims = index[tfidf[test_doc_vec]]
print(sims)

# 换个输出格式
for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    print(document_number, score)
