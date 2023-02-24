# -*- coding: utf-8 -*-
"""
    :author: Grey Li (李辉)
    :url: http://greyli.com
    :copyright: © 2018 Grey Li <withlihui@gmail.com>
    :license: MIT, see LICENSE for more details.
"""
from flask_bootstrap import Bootstrap5
from flask import Flask
from flask_moment import Moment
from flask_sqlalchemy import SQLAlchemy

app = Flask('tf_idf')
app.config.from_pyfile('settings.py')
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True

# db = SQLAlchemy(app)
bootstrap = Bootstrap5(app)
# moment = Moment(app)
#
# from tf_idf import views, errors, commands
from tf_idf import views
