import re

import jieba
from flask import render_template

from tf_idf import app
from tf_idf.forms import HelloForm
from tf_idf.sklearn_train import Classifier


@app.route('/', methods=['GET', 'POST'])
def index():
    similar_cases = []
    form = HelloForm()
    if form.validate_on_submit():
        case = form.case.data
        similar_cases = in_the_end(case)
    return render_template('index.html', form=form, cases_list=similar_cases)


def in_the_end(case):
    a = Classifier()
    text = re.sub(r"[^\u4e00-\u9fa5 ]+", '', case)  # 正则只匹配中文
    test_corpus = [" ".join(jieba.lcut(text))]  # 精简模式，返回一个列表类型的结果
    cases = a.generate_test_indices(test_corpus)
    return a.similar_cases(cases)
