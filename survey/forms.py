from flask_wtf import FlaskForm
from wtforms import StringField, RadioField, DateField
from wtforms.validators import DataRequired, Email

class ConsentForm(FlaskForm):
    name = StringField('Name', [DataRequired()])
    email = StringField('Email', [DataRequired(), Email()])
    date = DateField('Date', [DataRequired()])

class SectionOneForm(FlaskForm):
    likert = RadioField('content', [DataRequired()], choices=[(1, 'Strongly Disagree'), (2, 'Disagree'), (3, 'Neutral'), (4, 'Agree'), (5, 'Strongly Agree')], coerce=int)

class QuestionForm(FlaskForm):
    binary = RadioField('style', [DataRequired()], choices=[(1, 'A'), (2, 'B')], coerce=int)
    likert = RadioField('content', [DataRequired()], choices=[(1, 'Very Different'), (2, 'Different'), (3, 'Neutral'), (4, 'Similar'), (5, 'Very Similar')], coerce=int)
