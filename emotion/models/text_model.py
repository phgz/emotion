
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from pathlib import Path
from tensorflow.keras.models import model_from_json
from emotion.features.text.extract_text import remove_nonascii, clean_punct_digits, bert_encode 
from emotion import module_dir, root_dir

ARTIFACTS_DIR = Path(module_dir / "artifacts")
MODEL = f"{ARTIFACTS_DIR}/text_model.h5"


class TextModel():
    '''
    Deep neural network classifier using BERT embeddings
    '''
    def __init__(self):
        """
        json_file= open(MODEL, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self._model = model_from_json(loaded_model_json, custom_objects={'KerasLayer':hub.KerasLayer})
        self._model.load_weights(WEIGHTS)
        """
        self._model =  tf.keras.models.load_model(MODEL, custom_objects={'KerasLayer':hub.KerasLayer})

    def preprocess(self, text_str):
        cleaned_str = clean_punct_digits(remove_nonascii(text_str))
        encoding = bert_encode(cleaned_str)
        return encoding

    # Converts the classes to their assigned sentiment
    def to_sentiment(self, proba):
        sent_dict = {0 : 'negative', 
                     1 : 'neutral',
                     2 : 'positive'}

        return sent_dict[proba]

    # Makes the prediction on encoding
    def predict(self, encoding):
        preds = self._model.predict(encoding)
        sent = self.to_sentiment(np.argmax(preds))
        return sent