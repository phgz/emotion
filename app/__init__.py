import os

import dvc.api
from emotion.models.audio_model import AudioModel
from flask import Flask

# initialize the Flask application
app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ["FLASK_SECRET_KEY"]

with dvc.api.open(
        'emotion/artifacts/audio_model.pkl',
        repo='https://github.com/philipgaudreau/emotion',
        mode="rb"
        ) as fd:
    audio_model = AudioModel(fd)

from app import routes
