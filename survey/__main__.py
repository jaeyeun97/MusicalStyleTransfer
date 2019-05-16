import IPython
from . import create_app, db
from .survey import add_questions
from .models import Participant, Likert

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
    return [p.name for p in ps]
       

with app.app_context():
    print(get_participants())
    IPython.embed()
