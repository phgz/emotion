# extract features from audio clips

# import os
import glob
import pathlib
import time
from datetime import timedelta
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from emotion import root_dir

# audio directories
AUDIO_DIR = Path(root_dir / "data/raw/audio")
AUDIO_CLIPS_DIR = Path(root_dir / "data/interim/audio")
AUDIO_FEATURES_DIR = Path(root_dir / "data/processed/audio")

# feature extraction parameters
AGG = 'mean'
N_MFCCS = 40
LEN_SECS = 3


def extract_features_mean(audio_file, len_secs=3,
                           n_mfccs=40, rms=False, zrc=False):
    '''
        extract mean features from audio clip (full or part)
        audio_file : wav file 16000 sampling rate, mono
        len_secs   : length of clip to use for median features
        n_mfccs    : number of mel frequency ceptral coefficients
        rms        : extract rms (energy)
        zrc        : extract zero crossing rate
        returns    : audio_features: pd.Series containing median audio features
                     feature_names : name of extracted features
    '''
    samples, srate = \
        librosa.load(audio_file, sr=None)
    
    if len_secs != 'full':
        num_samples = srate * len_secs
        if num_samples > len(samples):
            samples = samples[-num_samples:]
    
    audio_features = None
    feature_names = []
    if rms:
        feature_names += ['rms']
        audio_features = \
            np.mean(librosa.feature.rms(y=samples).T, axis=0)
    if zrc:
        feature_names += ['zrc']
        zrc = \
            np.mean(librosa.feature.zero_crossing_rate(y=samples).T,
                    axis=0)
        if isinstance(audio_features, np.ndarray):
            audio_features = np.append(audio_features, zrc)
        else:
            audio_features = zrc
    
    feature_names += ['mfcc_' + str(x) for x in range(1, n_mfccs + 1)]
    S = librosa.feature.melspectrogram(y=samples,
                                   sr=srate, n_mels=128, n_fft=1200,
                                   fmax=8000, win_length=1200, hop_length=800)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfccs)
    mfccs = np.mean(mfccs, axis=1)
    if isinstance(audio_features, np.ndarray):
        audio_features = np.append(audio_features, mfccs)
    else:
        audio_features = mfccs
    return audio_features, feature_names


def extract_features_median(audio_file, len_secs=3,
                           n_mfccs=40, rms=False, zrc=False):
    '''
        extract median features from audio clip (full or part)
        audio_file : wav file 16000 sampling rate, mono
        len_secs   : length of clip to use for median features
        n_mfccs    : number of mel frequency ceptral coefficients
        rms        : extract rms (energy)
        zrc        : extract zero crossing rate
        returns    : audio_features: pd.Series containing median audio features
                     feature_names : name of extracted features
    '''
    samples, srate = \
        librosa.load(audio_file, sr=None)
    
    if len_secs != 'full':
        num_samples = srate * len_secs
        if num_samples > len(samples):
            samples = samples[-num_samples:]
    
    audio_features = None
    feature_names = []
    if rms:
        feature_names += ['rms']
        audio_features = \
            np.median(librosa.feature.rms(y=samples).T, axis=0)
    if zrc:
        feature_names += ['zrc']
        zrc = \
            np.median(librosa.feature.zero_crossing_rate(y=samples).T,
                    axis=0)
        if isinstance(audio_features, np.ndarray):
            audio_features = np.append(audio_features, zrc)
        else:
            audio_features = zrc
    
    feature_names += ['mfcc_' + str(x) for x in range(1, n_mfccs + 1)]
    S = librosa.feature.melspectrogram(y=samples,
                                   sr=srate, n_mels=64, #128,
                                   fmax=8000, hop_length=512)
    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfccs)
    mfccs = np.median(mfccs, axis=1)
    if isinstance(audio_features, np.ndarray):
        audio_features = np.append(audio_features, mfccs)
    else:
        audio_features = mfccs
    return audio_features, feature_names


def extract_features_from_files(files, agg='mean', len_secs=3, n_mfccs=40,
        rms=False, zrc=False, show_progress=True):
    '''
        extract audio features from files
        files  : directory containing wav files 16000 sampling rate, mono
        file_names : list of file names to extract features from
        agg        : type of aggregation to perform on features
                     'median' or 'mean'
        len_secs   : length of audio clip on which to extract features
        n_mfccs    : number of mel frequency ceptral coefficients
        rms        : extract rms (energy)
        zrc        : extract zero crossing rate
        returns    : audio_features: pd.Series containing median audio features
    '''
    audio_features = {}
    stime = time.time()
    num_files = len(files)
    for i, f in enumerate(files):
        if agg == 'median':
            clip_features, fnames = \
                extract_features_median(f, len_secs, n_mfccs, rms, zrc)
        else:
            clip_features, fnames = \
                extract_features_mean(f, len_secs, n_mfccs, rms, zrc)
        audio_features[str(i)] = clip_features

        if show_progress:
            if i % 10 == 0 and i != 0:
                print('.', end = '')
                if i % 500 == 0:
                    print(f" {i} de {num_files} fichiers")

    audio_features = pd.DataFrame(audio_features).T
    audio_features.columns = fnames
    etime = time.time()
    proc_time = timedelta(seconds = round(etime - stime))
    if show_progress:
        print("\n{audio_features.shape[0]}",
            f"fichiers extraits: {proc_time} (h:mm:ss)")
    return audio_features

def extract_features_from_dir(audio_dir, file_names=None,
                                    agg='mean', len_secs=3,
                                    n_mfccs=40, rms=False, zrc=False,
                                    show_progress=True):
    '''
        extract audio features from directory,
         all if file_names == None, files in file_names if not None
        audio_dir  : directory containing wav files 16000 sampling rate, mono
        file_names : list of file names to extract features from
                     file_id of form fileid_clipnumber without .wav
        agg        : type of aggregation to perform on features
                     'median' or 'mean'
        len_secs   : length of audio clip on which to extract features
        n_mfccs    : number of mel frequency ceptral coefficients
        rms        : extract rms (energy)
        zrc        : extract zero crossing rate
        returns    : audio_features: pd.Series containing median audio features
    '''
    audio_features = {}
    stime = time.time()
    if file_names is None:
        file_names = glob.glob(f"{audio_dir}/*.wav")
    else:
        file_names = [f"{audio_dir}/{f}.wav" for f in file_names]
    num_files = len(file_names)
    for i, f in enumerate(file_names):
        if agg == 'median':
            clip_features, fnames = \
                extract_features_median(f, len_secs, n_mfccs, rms, zrc)
        else:
            clip_features, fnames = \
                extract_features_mean(f, len_secs, n_mfccs, rms, zrc)
        clip_id = f.rsplit('.', maxsplit = 1)[0].rsplit('/', maxsplit = 1)[-1]
        audio_features[clip_id] = clip_features

        if show_progress:
            if i % 10 == 0 and i != 0:
                print('.', end = '')
                if i % 500 == 0:
                    print(f" {i} de {num_files} fichiers")

    audio_features = pd.DataFrame(audio_features).T
    audio_features.columns = fnames
    etime = time.time()
    proc_time = timedelta(seconds = round(etime - stime))
    if show_progress:
        print("\n{audio_features.shape[0]}",
            f"fichiers extraits: {proc_time} (h:mm:ss)")
    return audio_features

def main():
    feature_file = \
        Path(AUDIO_FEATURES_DIR / \
            "audio_features.csv")
    if pathlib.Path.is_file(feature_file):
        print(f"{feature_file} already exists, no export to csv")
    else:
        features = \
            extract_features_from_dir(AUDIO_CLIPS_DIR, file_names=None,
                    agg=AGG, len_secs=LEN_SECS, n_mfccs=N_MFCCS
            )
        features.to_csv(feature_file,
                header=True, index=True)
        print(f"features exported to {feature_file}")

if __name__ == "__main__":
    main()
