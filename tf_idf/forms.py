# -*- coding: utf-8 -*-

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length


class HelloForm(FlaskForm):
    case = StringField('case', validators=[DataRequired(), Length(1, 800)])
    submit = SubmitField()
