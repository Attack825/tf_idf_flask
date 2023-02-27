# -*- coding: utf-8 -*-

from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length


class HelloForm(FlaskForm):
    case = TextAreaField('case', validators=[DataRequired(), Length(1, 1200)])
    submit = SubmitField()
