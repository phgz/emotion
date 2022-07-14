from emotion.models.text_model import TextModel


def test_text_model_local():
    text_model = TextModel()
    processed_features = text_model.preprocess(["I fear for my life, oh.", "Hello world"])
    predictions = text_model.predict(processed_features)
    assert len(predictions) == 2
