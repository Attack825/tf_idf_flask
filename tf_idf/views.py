from flask import flash, redirect, url_for, render_template
from tf_idf import app
from tf_idf.forms import HelloForm


@app.route('/', methods=['GET', 'POST'])
def index():
    form = HelloForm()
    if form.validate_on_submit():
        case = form.case.data
        return redirect(url_for('index'))
    return render_template('index.html', form=form)
