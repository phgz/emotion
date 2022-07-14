
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from pathlib import Path
from tensorflow.keras.models import model_from_json
from emotion.features.text.extract_text import remove_nonascii, clean_punct_digits, bert_encode, remove_stamps_str
from emotion import module_dir, root_dir

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ARTIFACTS_DIR = Path(module_dir / "artifacts")

MODEL = f"{ARTIFACTS_DIR}/text_model.json"
WEIGHTS =f"{ARTIFACTS_DIR}/weights.h5"
class TextModel():
    '''
    Deep neural network classifier using BERT embeddings
    '''
    def __init__(self):
        
        json_file= open(MODEL, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self._model = model_from_json(loaded_model_json, custom_objects={'KerasLayer':hub.KerasLayer})
        self._model.load_weights(WEIGHTS)
        
        #self._model =  tf.keras.models.load_model(MODEL, custom_objects={'KerasLayer':hub.KerasLayer})

    def preprocess(self, texts):

       

        cleaned_list = [clean_punct_digits(remove_nonascii(text)) for text in texts]
        encoding = bert_encode(cleaned_list)
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
        sents = self.to_sentiment([np.argmax(pred) for pred in preds])
        return sents
