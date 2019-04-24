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
    section = Column(Integer)
    sample_num = Column(Integer)
    exp_first = Column(String)
    exp_second = Column(String)


class Binary(db.Model):
    __tablename__ = 'binaries'
    id = Column(Integer, primary_key=True)
    participant_id = Column(Integer, ForeignKey('participants.id'))
    participant = relationship('Participant', backref=backref('binaries'))
    question_id = Column(Integer, ForeignKey('questions.id'))
    question = relationship('Question', backref=backref('binaries'))
    answer = Column(Boolean)  # True: Experiment 1, False: Experiment 2
    created_at = Column(DateTime)


class Likert(db.Model):
    __tablename__ = 'likerts'
    id = Column(Integer, primary_key=True)
    participant_id = Column(Integer, ForeignKey('participants.id'))
    participant = relationship('Participant', backref=backref('likerts'))
    question_id = Column(Integer, ForeignKey('questions.id'))
    question = relationship('Question', backref=backref('likerts'))
    answer = Column(Integer)  # 1 ~ 10
    created_at = Column(DateTime)
