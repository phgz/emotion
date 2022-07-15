import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub
from emotion import module_dir, root_dir
from emotion.features.text.extract_text import (
    bert_encode,
    clean_punct_digits,
    remove_nonascii,
    remove_stamps_str
)
from tensorflow.keras.models import model_from_json


ARTIFACTS_DIR = Path(module_dir / "artifacts")

MODEL = f"{ARTIFACTS_DIR}/text_model.json"
WEIGHTS =f"{ARTIFACTS_DIR}/weights.h5"

class TextModel():
    '''
    Deep neural network classifier using BERT embeddings
    '''
    def __init__(self, model=None, weights=None):
            self._model = model_from_json(model and open(model, 'r').read() or open(MODEL, 'r').read(), custom_objects={'KerasLayer':hub.KerasLayer})
            self._model.load_weights(weights or WEIGHTS)
            

    def preprocess(self, texts):
        cleaned_list = [clean_punct_digits(remove_nonascii(text)) for text in texts]
        #encoding = bert_encode(cleaned_list)
        encoding = tf.constant(cleaned_list)
        return encoding

    # Converts the classes to their assigned sentiment
    def to_sentiment(self, proba_list):
        sent_dict = {0 : 'negative', 
                     1 : 'neutral',
                     2 : 'positive'}

        return [sent_dict[proba] for proba in proba_list]

    # Makes the prediction on encoding
    def predict(self, encoding):
        preds = self._model.predict(encoding)
        sents = self.to_sentiment([np.random.randint(3) for i in range(len(encoding))])
        #sents = self.to_sentiment([np.argmax(pred) for pred in preds])
        return sents