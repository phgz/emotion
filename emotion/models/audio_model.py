# load model, scaler and preprocessing

import pickle
from pathlib import Path

from emotion import module_dir
from emotion.features.audio.extract_features import extract_features_from_files

ARTIFACTS_DIR = Path(module_dir / "artifacts")
MODEL = f"{ARTIFACTS_DIR}/audio_model.pkl"

class AudioModel():
    '''
        SVM audio sentiment classifier
    '''
    def __init__(self, model=None):
        self._model = pickle.load(model or open(MODEL, "rb"))
        self.class_names = self._model.class_names

    def preprocess(self, files):
        features = \
                extract_features_from_files(files=files,
                        agg='mean', len_secs=3, n_mfccs=40,
                        show_progress=False
        )
        features = self._model.scaler.transform(features)
        return features

    def predict(self, features):
        predictions = self._model.predict(features)
        predictions = [self.class_names[p] for p in predictions]

        return(predictions)
