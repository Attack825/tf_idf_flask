# -*- coding: utf-8 -*-
from flask_bootstrap import Bootstrap5
from flask import Flask

app = Flask('tf_idf')
app.config.from_pyfile('settings.py')
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True

bootstrap = Bootstrap5(app)

from tf_idf import views

