
import pickle
import numpy as np
import extract_text
import tensorflow_hub as hub
from pathlib import Path
from tensorflow.keras.models import model_from_json
from emotion.features.text.extract_text import remove_non_ascii, clean_stopwords_digits, bert_encode 



class TextModel():
    '''
    Deep neural network classifier using BERT embeddings
    '''
    def __init__(self):
        json_file= open("model.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()

        self._model = model_from_json(loaded_model_json, custom_objects={'KerasLayer':hub.KerasLayer})
        self._model.load_weights("model.h5")

    def preprocess(self, text_str):
        cleaned_str = clean_stopwords_digits(remove_non_ascii(text_str))
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
        



    