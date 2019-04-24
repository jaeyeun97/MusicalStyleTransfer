import IPython
from . import create_app

app = create_app()
with app.app_context():
    IPython.embed()
