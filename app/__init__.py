import os
from pathlib import Path


def iterpath(path):
    print(path)
    for p in path.iterdir():
        print(p)
    print("-----------------")

print(os.getenv("LD_LIBRARY_PATH"))
path = Path("/")
iterpath(path)
path = path / "app"
iterpath(path)
path = path / ".apt"
iterpath(path)
path = path / "usr"
iterpath(path)
path = path / "lib"
iterpath(path)
path = path / "x86_64-linux-gnu"
iterpath(path)
iterpath(Path("/usr/lib/x86_64-linux-gnu"))


import dvc.api
# from emotion.models.audio_model import AudioModel
from flask import Flask

# initialize the Flask application
app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ["FLASK_SECRET_KEY"]

# with dvc.api.open(
#         'emotion/artifacts/audio_model.pkl',
#         repo='https://github.com/philipgaudreau/emotion',
#         mode="rb"
#         ) as fd:
#     audio_model = AudioModel(fd)

from app import routes
