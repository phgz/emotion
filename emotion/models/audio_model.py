# load model, scaler and preprocessing

import pickle
from pathlib import Path

from emotion import module_dir, root_dir
from emotion.features.audio.extract_features import extract_features_from_dir

# from features.audio.extract_features import extract_features_mean

ARTEFACTS_DIR = Path(root_dir / "artifacts")
MODEL = f"{ARTEFACTS_DIR}/audio_model.pkl"
SCALER = f"{ARTEFACTS_DIR}/audio_scaler.pkl"
# MODEL = Path(ARTEFACTS_DIR / "audio_model.pkl")
# SCALER = Path(ARTEFACTS_DIR / "audio_scaler.pkl")

class AudioModel():
    def __init__(self):
        # print(MODEL)
        # print(SCALER)
        with open(MODEL, "rb") as f:
            self._model = pickle.load(f)
        # with open(SCALER, "rb") as f:
        #     self._scaler = pickle.load(f)
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
