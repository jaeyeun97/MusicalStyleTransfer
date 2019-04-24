import os
import itertools
from random import shuffle
from functools import wraps
from flask import Blueprint, flash, g, redirect, render_template, request, session, url_for
from . import db
from .models import Question, Participant, Binary, Likert
from .forms import ConsentForm, QuestionForm, SectionOneForm

bp = Blueprint('survey', __name__, url_prefix='/survey')

def after_consent(f):
    @wraps(f)
    def func(*args, **kwargs):
        if 'participant_id' not in session:
            return redirect(url_for('.start'))
        return f(*args, **kwargs)
    return func

def get_sample_name(exp, sample_num=0):
    sample_name = '{:03d}'.format(sample_num)
    return os.path.abspath(os.path.join('results', exp, sample_name))

def get_file_paths(exp_first, exp_second, sample_num):
    second_A = os.path.join(get_sample_name(exp_second, sample_num), 'fake_B.wav')
    if 'real' in exp_first:
        first_A = os.path.join(get_sample_name(exp_second, sample_num), 'real_A.wav')
    else:
        first_A = os.path.join(get_sample_name(exp_first, sample_num), 'fake_B.wav')
    return first_A, second_A

def get_new_question(section, exp_first, exp_second, sample_num):
    return Question(exp_first=exp_first,
                    exp_second=exp_second,
                    sample_num=sample_num,
                    section=section)

def add_questions(section, limit, *exps):
    questions = list()
    for i in range(limit):
        for exp in exps:
            questions.append(get_new_question(section, 'real', exp, i))
        for exp in itertools.combinations(exps, 2):
            questions.append(get_new_question(section, exp[0], exp[1], i))
    shuffle(questions)
    db.session.add_all(questions)
    db.session.commit()

def get_questions(section_num):
    bs = Binary.query.filter(Binary.participant_id == session['participant_id']).all()
    answered = set(b.question_id for b in bs)
    qs = Question.query.filter(Question.section == section_num).all()
    qids = set(q.id for q in qs)
    return list(qids - answered)

def get_participant():
    return Participant.query.filter(Participant.id == session['participant_id']).first()

@bp.route('/', methods=['GET', 'POST'])
def start():
    consentForm = ConsentForm()
    if consentForm.validate_on_submit():

        user = Participant.query.filter(Participant.email == consentForm.email.data).first()
        if user is None:
            user = Participant(name=consentForm.name.data,
                               email=consentForm.email.data,
                               date=consentForm.date.data)
            db.session.add(user)
            db.session.commit()
        session['participant_id'] = user.id
        return redirect(url_for('.section_one'))
    return render_template('start.html', form=consentForm)

@bp.route('/section1', methods=['GET', 'POST'])
@after_consent
def section_one():
    session['section'] = 1
    session['questions'] = get_questions(1)

    form = SectionOneForm()
    if form.validate_on_submit():
        chillness = Likert(participant_id=session['participant_id'],
                           question_id=None,
                           answer=form.likert.data)
        db.session.add(chillness)
        db.session.commit()
        q_id = session['questions'][0]
        return redirect(url_for('.render_question', q_id=q_id))

    return render_template('section1.html', form=form)

@bp.route('/section2')
@after_consent
def section_two():
    session['section'] = 2
    session['questions'] = get_questions(2)
    return render_template('section2.html',
                           next=url_for('.render_question', q_id=session['questions'][0]))

@bp.route('/thankyou')
@after_consent
def thank_you():
    return render_template('thankyou.html')

@bp.route('/<int:q_id>', methods=['GET', 'POST'])
@after_consent
def render_question(q_id):
    form = QuestionForm()
    print(session['questions'])
    if form.validate_on_submit():
        session['questions'].remove(q_id)
        session.modified = True
        # save results
        style = Binary(participant_id=session['participant_id'],
                       question_id=q_id,
                       answer=True if form.binary.data == 1 else False)
        content = Likert(participant_id=session['participant_id'],
                         question_id=q_id,
                         answer=form.likert.data)
        db.session.add(style)
        db.session.add(content)
        db.session.commit()
        if len(session['questions']) == 0:
            if session['section'] == 1:
                return redirect(url_for('.section_two'))
            else:
                return redirect(url_for('.thank_you'))
        else:
            return redirect(url_for('.render_question', q_id=session['questions'][0]))
    else:    
        question = Question.query.filter(Question.id == q_id).first()
        paths = get_file_paths(question.exp_first, question.exp_second, question.sample_num)
        return render_template('question.html', form=form, paths=paths)
