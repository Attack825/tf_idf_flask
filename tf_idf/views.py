from flask import render_template
from tf_idf import app
from tf_idf.forms import HelloForm


@app.route('/', methods=['GET', 'POST'])
def index():
    similar_cases = []
    form = HelloForm()
    if form.validate_on_submit():
        case = form.case.data
        similar_cases = shuchu(case)
    return render_template('index.html', form=form, cases_list=similar_cases)


def shuchu(case):
    return [case, case, case, case, case]
