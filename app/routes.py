from flask import flash, redirect, render_template, request

from app import app, audio_model
from app.utils import prepare_data, print_items


@app.route('/index')
def index():
   return render_template('index.html')


@app.route("/index", methods=["POST"])
def predict():
    files = request.files.getlist('files[]')

    try:
        audio_files, (text_files, text_files_texts) = prepare_data(files)

    except ValueError as e:
        flash(str(e))

        return redirect("/index")

    predictions_audio = audio_files and audio_model.predict(audio_model.preprocess([file.stream for file in audio_files])) or []
    predictions_texte = text_files and [{line: str(i) for i, line in enumerate(texts)} for texts in text_files_texts] or []

    flash(print_items({"predictions_audio":{file.filename: pred for file, pred in zip(audio_files, predictions_audio)}, "predictions_texte":{file.filename: pred for file, pred in zip(text_files, predictions_texte)}}))

    return redirect("/index")
