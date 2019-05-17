import IPython
from . import create_app, db
from .survey import add_questions
from .models import Participant, Likert, Question

app = create_app()

def count(ls):
    result = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for x in map(lambda x: x.answer, ls):
        result[x] += 1
    return result

def aggregate_results(section_num):
    ls = Likert.query.filter(Likert.question.has(section = section_num)).all()
    contents = [l for l in ls if l.is_content]
    styles = [l for l in ls if not l.is_content]
    
    print('Content results')
    raw_content = count(contents)
    perc_content = {k: v / len(contents) for k, v in raw_content.items()}
    print(raw_content)
    print(perc_content)
    print('Style bins')
    raw_styles = count(styles)
    perc_styles = {k: v / len(styles) for k, v in raw_styles.items()}
    print(raw_styles)
    print(perc_styles) 

def get_participants():
    ps = Participant.query.all()
    print('Length: {}'.format(len(ps))) 
    print([p.name for p in ps])
    return ps
       
def export(file_name):
    musicians = ['Jake Moscrop', 'Victoria Clarkson', 'Keir Lewis', 'Trojan Nakade', 'Kaamya Varagur', 'William Collins', 'William Debnam', 'Amanda McHugh']
     
    with open(file_name, 'w') as f:
        f.write('participant_id,is_musician,section,question_id,sample_num,qtype,answer\n')
        answers = Likert.query.all()
        for answer in answers:
            question = answer.question
            section = question.section
            is_musician = False
            if answer.participant.name in musicians:
                is_musician = True
            qtype = 'content' if answer.is_content else 'style'
            f.write('{participant_id},{is_musician},{section},{question_id},{sample_num},{qtype},{answer}\n'.format(
                participant_id=answer.participant_id,
                is_musician=is_musician,
                section=section,
                question_id=question.id,
                sample_num=question.sample_num,
                qtype=qtype,
                answer=answer.answer
            ))

with app.app_context():
    ps = get_participants()
    IPython.embed()
