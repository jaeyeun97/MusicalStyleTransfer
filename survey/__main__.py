import IPython
from . import create_app, db
from .survey import add_questions

app = create_app()

with app.app_context():
    IPython.embed()
