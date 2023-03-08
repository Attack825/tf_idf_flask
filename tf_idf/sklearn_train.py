# -*- coding:utf-8 -*-
import re

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

train_filename = 'data10.json'
test_filename = ''


# 数据预处理
def generate_corpus(filename):
    text_corpus = []
    with open(filename, 'r', encoding='utf8') as f:
        m = f.readlines()
        for i in m:
            c = eval(i)  # 去除双引号,字符串变成字典
            text = re.sub(r"[^\u4e00-\u9fa5 ]+", '', c['fullText'])  # 正则只匹配中文
            data_list = " ".join(jieba.lcut(text))  # 精简模式，返回一个列表类型的结果
            text_corpus.append(data_list)
        return text_corpus


train_corpus = generate_corpus(train_filename)
# print(train_corpus)  # 返回训练预料

# 读取停顿词列表
stopword_list = [k.strip() for k in open('stopwords', encoding='utf8').readlines() if k.strip() != '']

# 模型建立
tfidf_model = TfidfVectorizer(stop_words=stopword_list, max_features=None, max_df=0.9)
# 默认去除只有一个字的词，可修改token_pattern, max_df修改区间如果在所有文章出现概率高于0.9则剔除
# max_features：默认为None，可设为int，对所有关键词的term frequency进行降序排序，只取前max_features个作为关键词集。
tfidf = tfidf_model.fit_transform(train_corpus)  # 输入便转化得到tf-idf矩阵，稀疏矩阵表示法
print(tfidf.toarray())  # 矩阵标准化
print(tfidf_model.get_feature_names_out())

# 求余弦相似度
cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
# print(cosine_similarities)

# 相关文档索引
related_docs_indices = cosine_similarities.argsort()[:-6:-1]  # 选择相似度最高的5个
print(related_docs_indices)
print(cosine_similarities[related_docs_indices])
