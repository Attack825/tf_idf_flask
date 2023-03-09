# -*- coding:utf-8 -*-
import pickle
import re

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

train_filename = 'D:\\pythonProject\\tf_idf_flask\\tf_idf\\data10.json'
test_filename = 'D:\\pythonProject\\tf_idf_flask\\tf_idf\\data1.json'
train_pickle_path = 'D:\\pythonProject\\tf_idf_flask\\tf_idf\\train.pickle'
train_features_pickle_path = 'D:\\pythonProject\\tf_idf_flask\\tf_idf\\train_comment_features.pickle'
stopwords_path = 'D:\\pythonProject\\tf_idf_flask\\tf_idf\\stopwords.txt'
stopword_list = [k.strip() for k in open(stopwords_path, encoding='utf8').readlines() if k.strip() != '']


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


class Classifier:
    def __init__(self):
        self.train_model = self.train_model()

    # 数据预处理

    # 根据索引输出文章列表
    def similar_cases(self, indices):
        similar_cases = []
        with open(train_filename, 'r', encoding='utf8') as f:
            for i, line in enumerate(f):
                if i in indices:
                    c = eval(line)  # 去除双引号,字符串变成字典
                    similar_cases.append(c['fullText'])
        return similar_cases

    def train_model(self):

        tfidf_model = TfidfVectorizer(stop_words=stopword_list, max_features=None, max_df=0.9)
        return tfidf_model

    def generate_train_matrx(self, corpus):
        tfidf_model = self.train_model
        # 默认去除只有一个字的词，可修改token_pattern, max_df修改区间如果在所有文章出现概率高于0.9则剔除
        # max_features：默认为None，可设为int，对所有关键词的term frequency进行降序排序，只取前max_features个作为关键词集。
        tfidf = tfidf_model.fit(corpus)  # 输入便转化得到tf-idf矩阵，稀疏矩阵表示法
        vectorizer = tfidf_model.transform(corpus)
        # 保存经过fit的tfidfvectorizer预测时使用
        pickle.dump(tfidf, open(train_pickle_path, 'wb'))
        pickle.dump(vectorizer, open(train_features_pickle_path, "wb"))
        return tfidf

    # train_tfidf = generate_model(train_corpus)  # 稀疏矩阵
    # print(test_corpus)

    def generate_test_indices(self, corpus):  # 运用保存的特征返回相似度高的索引
        # 加载TfidfVectorizer
        train_pickle = pickle.load(open(train_pickle_path, 'rb'))
        train_tfidf_load = pickle.load(open(train_features_pickle_path, "rb"))
        test_tfidf = train_pickle.transform(corpus)
        # test_tfidf.vocabulary = tfidf_model.vocabulary_
        # 求余弦相似度
        cosine_similarities = linear_kernel(test_tfidf, train_tfidf_load).flatten()
        # 相关文档索引
        related_docs_indices = cosine_similarities.argsort()[:-6:-1]  # 选择相似度最高的5个
        return related_docs_indices


if __name__ == '__main__':
    a = Classifier()
    # train_corpus = generate_corpus(train_filename)
    test_corpus = generate_corpus(test_filename)
    # a.generate_train_matrx(train_corpus)
    index = a.generate_test_indices(test_corpus)
    print(a.similar_cases(index))
# index = generate_test_indices(test_corpus, 'train.pickle')
# print(train_corpus)  # 返回训练预料
# print(tfidf.toarray())  # 矩阵标准化
# print(tfidf_model.get_feature_names_out())
# print(cosine_similarities)
# print(related_docs_indices)
# print(cosine_similarities[related_docs_indices])
# print(index)
