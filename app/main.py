import tempfile
from pathlib import Path

import dvc.api
from emotion.models.audio_model import AudioModel
from flask import Flask, flash, jsonify, redirect, render_template, request

from app.utils import allowed_file

# initialize the Flask application
app = Flask(__name__)
# Pull the output of the DVC stage used to generate the serialized model when running on a
# deployed server. For example:

with dvc.api.open(
        'emotion/artifacts/audio_model.pkl',
        repo='https://github.com/philipgaudreau/emotion',
        rev="feature/audio-features",
        mode="rb"
        ) as fd:
        audio_model = AudioModel(fd)

@app.route('/')
def upload_form():
   return render_template('upload.html', title="Accueil", mod={"titre": "Placement de plateforme"})

@app.route("/", methods=["POST"])
def predictAudio():
    """
    Accepts a payload w/content-type="application/json" and expects a `data` key/value.
    `data` is passed into the model for inference.
    """
    
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        print(files)

        for file in files:
            if not allowed_file(file.filename):
                return f"The extension for {file.filename} is not allowed."

        text_files, audio_files = text_audio = ([], [])

        for file in files:
            text_audio[file.filename.rsplit('.', 1)[1].lower() == "wav"].append(file)

        predictions_audio = audio_model.predict(audio_model.preprocess([file.stream for file in audio_files]))

        return jsonify({"predictions_audio":{file.filename:pred for file, pred in zip(audio_files, predictions_audio)}}, 200)

if __name__ == "__main__":
    app.run(port=8888)
