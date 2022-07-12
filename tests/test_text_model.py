from emotion import root_dir
from emotion.models.text_model import TextModel


def test_audio_model():
    text_model = TextModel()
    processed_features = text_model.preprocess([root_dir / "data/raw_sample/text/UlTJmndbGHM.txt"])
    predictions = text_model.predict(processed_features)
    assert len(predictions) == 1
