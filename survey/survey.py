import os
import itertools
from random import shuffle
from functools import wraps
from flask import Blueprint, flash, g, redirect, render_template, request, session, url_for, send_from_directory
from . import db
from .models import Question, Participant, Likert
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
    return os.path.join(exp, sample_name)

def get_file_paths(exp, sample_num, btoa):
    dom_A = 'B' if btoa else 'A'
    dom_B = 'A' if btoa else 'B'
    real_A = os.path.join(get_sample_name(exp, sample_num), 'real_{}.0.wav'.format(dom_A)) 
    # real_B = os.path.join(get_sample_name(exp, sample_num), 'real_{}.0.wav'.format(dom_B))
    fake_B = os.path.join(get_sample_name(exp, sample_num), 'fake_{}.0.wav'.format(dom_B))
    return real_A, fake_B

def get_new_question(section, exp, sample_num, BtoA):
    return Question(exp=exp,
                    sample_num=sample_num,
                    section=section,
                    BtoA=BtoA)

def add_questions(section, limit, exp, BtoA):
    questions = list()
    for i in range(limit):
        questions.append(get_new_question(section, exp, i, BtoA)) 
    shuffle(questions)
    db.session.add_all(questions)
    db.session.commit()

def get_questions(section_num):
    ls = Likert.query.filter(Likert.participant_id == session['participant_id']).all()
    answered = set(l.question_id for l in ls)
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

# @bp.route('/section1', methods=['GET', 'POST'])
# @after_consent
# def section_one():
#     session['section'] = 1
#     session['questions'] = get_questions(1)
# 
#     if Likert.query.filter(Likert.question_id == None).first() is not None:
#         if session['questions']:
#             q_id = session['questions'][0]
#             return redirect(url_for('.render_question', q_id=q_id))
#         return redirect(url_for('.section_two'))
#     form = SectionOneForm()
#     if form.validate_on_submit():
#         chillness = Likert(participant_id=session['participant_id'],
#                            question_id=None,
#                            answer=form.likert.data)
#         db.session.add(chillness)
#         db.session.commit() 
#     return render_template('section1.html', form=form)

@bp.route('/section1')
@after_consent
def section_one():
    session['section'] = 1
    session['questions'] = get_questions(1)

    if session['questions']:
        return render_template('section1.html',
                               next=url_for('.render_question', q_id=session['questions'][0]))
    else:
        return redirect(url_for('.section_two'))

@bp.route('/section2')
@after_consent
def section_two():
    session['section'] = 2
    session['questions'] = get_questions(2)

    if session['questions']:
        return render_template('section2.html',
                               next=url_for('.render_question', q_id=session['questions'][0]))
    else:
        return redirect(url_for('.thank_you'))



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
        style = Likert(participant_id=session['participant_id'],
                       question_id=q_id,
                       is_content=False,
                       answer=form.style.data)
        content = Likert(participant_id=session['participant_id'],
                         question_id=q_id,
                         is_content=True,
                         answer=form.content.data)
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
        paths = get_file_paths(question.exp, question.sample_num, question.BtoA)
        return render_template('question.html', form=form, paths=paths, section=session['section'])

@bp.route('/results/<path:path>')
def get_results(path):
    return send_from_directory(os.path.abspath('results'), path, mimetype='audio/wav')
