from . import db
from sqlalchemy import Column, Integer, Boolean, ForeignKey, String, DateTime
from sqlalchemy.orm import backref, relationship


class Participant(db.Model):
    __tablename__ = 'participants'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    date = Column(DateTime)


class Question(db.Model):
    __tablename__ = 'questions'
    id = Column(Integer, primary_key=True)
    BtoA = Column(Boolean, default=False)
    section = Column(Integer)
    sample_num = Column(Integer)
    exp = Column(String)


class Likert(db.Model):
    __tablename__ = 'likerts'
    id = Column(Integer, primary_key=True)
    participant_id = Column(Integer, ForeignKey('participants.id'))
    participant = relationship('Participant', backref=backref('likerts'))
    question_id = Column(Integer, ForeignKey('questions.id'))
    question = relationship('Question', backref=backref('likerts'))
    is_content = Column(Boolean) # True: Content Q, False: style Q
    answer = Column(Integer)  # 1 ~ 10
    created_at = Column(DateTime)
