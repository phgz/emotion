# load model, scaler and preprocessing

import pickle
from pathlib import Path

from emotion import module_dir
from emotion.features.audio.extract_features import extract_features_from_dir

ARTIFACTS_DIR = Path(module_dir / "artifacts")
MODEL = f"{ARTIFACTS_DIR}/audio_model.pkl"

class AudioModel():
    '''
        SVM audio sentiment classifier
    '''
    def __init__(self):
        with open(MODEL, "rb") as f:
            self._model = pickle.load(f)
        self.class_names = self._model.class_names

    def preprocess(self, audio_dir, file_names):
        features = \
                extract_features_from_dir(audio_dir,
                        file_names=file_names,
                        agg='mean', len_secs=3, n_mfccs=40,
                        show_progress=False
        )
        features = self._model.scaler.transform(features)
        return features

    def predict(self, features):
        predictions = self._model.predict(features)
        return(predictions)
