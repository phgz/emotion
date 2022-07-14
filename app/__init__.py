import os
import tempfile
from pathlib import Path

import dvc.api
from emotion.models.audio_model import AudioModel
from emotion.models.text_model import TextModel
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

with dvc.api.open(
        'emotion/artifacts/text_model.json',
        repo='https://github.com/philipgaudreau/emotion',
        mode="rt",
        rev="feature/text-model"
        ) as arch_fd:
    with dvc.api.open(
            'emotion/artifacts/weights.h5',
            repo='https://github.com/philipgaudreau/emotion',
            mode="rb",
            rev="feature/text-model"
            ) as weights_fd:
        with tempfile.TemporaryDirectory() as td:
                model_path = Path(td) / "text_model.json"
                weights_path = Path(td) / "weights.h5"
                open(model_path, "wt").write(arch_fd.read())
                open(weights_path, "wb").write(weights_fd.read())
                text_model = TextModel(arch_fd, weights_fd)

from app import routes
