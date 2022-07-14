import os
import subprocess

from emotion import module_dir
from emotion.models.audio_model import AudioModel
from emotion.models.text_model import TextModel
from flask import Flask

# initialize the Flask application
app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ["FLASK_SECRET_KEY"]

subprocess.run(["dvc", "get", "-o", module_dir, "https://github.com/philipgaudreau/emotion", "emotion/artifacts"])

audio_model = AudioModel()
text_model = TextModel()

from app import routes
