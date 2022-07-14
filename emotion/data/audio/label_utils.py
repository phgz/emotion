# utils to get labels for audio model

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from emotion import root_dir

LABELS_DIR = Path(root_dir / "data/raw/labels")
AUDIO_CLIPS_DIR = Path(root_dir / "data/interim/audio")
OUT_DIR = Path(root_dir / "data/processed/audio")

def load_all_ratings(labels_dir, split_id_clip = False):
    '''
        loads all ratings
        - replaces sentiment by positive if sentiment > 0
          or by negative if sentiment < 0
        - replaces all emotions by 1 if > 0
        - adds none = 1 where all emotions + sentiment == 0
        labels_dir    : directory containing labels (ratings)
        split_id_clip : if true id and clip columns separate
                        if false single id_clip column
        returns       : DataFrame containing all ratings (3 per text/clip)
    '''
    
    label_files = glob.glob(f"{labels_dir}/*.csv")
    df_labels = []

    for filename in label_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        df_labels.append(df)


    df_labels = pd.concat(df_labels, axis=0, ignore_index=True)

    label_cols = ['Input.VIDEO_ID', 'Input.CLIP',
              'Answer.anger', 'Answer.disgust',
              'Answer.fear', 'Answer.happiness',
              'Answer.sadness', 'Answer.surprise',
              'Answer.sentiment']

    label_new_cols = ['id', 'clip',
                      'anger', 'disgust',
                      'fear', 'happiness',
                      'sadness', 'surprise',
                      'sentiment']
    df_labels = df_labels[label_cols]
    df_labels.columns = label_new_cols

    # drop row all nan
    isna_idx = \
        df_labels.index[df_labels[df_labels.columns[2:]].isna().all(axis=1)]
    df_labels.drop(index=isna_idx, inplace=True)
    # replace remaining nan's with 0
    df_labels = df_labels.replace({np.nan : 0})
    # convert ratings to int
    df_labels[label_new_cols[2:]] = df_labels[label_new_cols[2:]].astype('Int64')
    # set emotions to 0 or 1
    df_labels[label_new_cols[2:-1]] = \
        df_labels[label_new_cols[2:-1]].applymap(lambda x : 1 if x > 0 else 0)

    # if sentiment > 0 convert to positive = 1, elif < 0 convert to negative = 1
    #  if none of emotion or sentiment == 1, set none to 1
    df_labels['positive'] = \
        df_labels['sentiment'].map(lambda x : 1 if x > 0 else 0)
    df_labels['negative'] = \
        df_labels['sentiment'].map(lambda x : 1 if x < 0 else 0)

    # drop sentiment column (now in positive/negative)
    df_labels.drop(columns='sentiment', inplace=True)
    
    df_labels['none'] = 0
    none_idx = \
        df_labels[df_labels[df_labels.columns[2:]].sum(axis=1) == 0].index

    df_labels.loc[none_idx,'none'] = 1

    label_new_cols = ['id', 'clip',
                      'anger', 'disgust',
                      'fear', 'happiness',
                      'sadness', 'surprise',
                      'sentiment']
 
    # remove '/' from id's
    df_labels['id'] = df_labels['id'].map(lambda x : str(x).split("/")[-1])
    if not split_id_clip:
        df_labels['id'] = df_labels['id'] + '_' + df_labels['clip'].astype(str)
        df_labels.drop(columns = 'clip', inplace = True)
        
        label_new_cols = ['id',
                          'none', 'positive', 'negative',
                          'anger', 'disgust',
                          'fear', 'happiness',
                          'sadness', 'surprise'
                         ]
    else:
        label_new_cols = ['id', 'clip',
                          'none', 'positive', 'negative',
                          'anger', 'disgust',
                          'fear', 'happiness',
                          'sadness', 'surprise'
                         ]
        
    df_labels = df_labels[label_new_cols]
    return df_labels

def aggregate_ratings(ratings):
    '''
        aggregate labels to 1 if 2+ ratings of 3 aggree
        ratings : pd.DataFrame containing ratings to aggregate
        returns : pd.DataFrame with aggregated ratings (labels)
    '''
    grp_labels = ratings.groupby('id').sum()
    # label to 1 if > 1 else 0
    grp_labels = grp_labels.applymap(lambda x : 1 if x > 1 else 0)
    # drop rows where all == 0
    idx = grp_labels[grp_labels.sum(axis =1) == 0].index
    grp_labels.drop(index = idx, inplace=True)
    print(f"{len(idx)} rows dropped")
    print(f"{grp_labels.shape[0]} grouped labels")
    return grp_labels

def get_sentiment_labels(labels_dir, audio_dir):
    '''
        get labels having none, positive or negative == 1
         having audio clips labels_dir : directory containing ratings (labels)
        audio_dir  : directory containing audio clips
        returns    : pd.DataFrame with aggregated labels
                     columns: [id, none, positive, negative]
    '''
    all_ratings = load_all_ratings(labels_dir)
    # keep ratings having positive, negative or none
    mask_positive = all_ratings['positive'] == 1
    mask_negative = all_ratings['negative'] == 1
    mask_none = all_ratings['none'] == 1
    sentiment_ratings = \
        all_ratings[(mask_positive) | (mask_negative) | (mask_none)]\
            [['id','none','positive','negative']].copy()
    # aggregate ratings
    sentiment_labels = aggregate_ratings(sentiment_ratings)
    # get audio clip names
    audio_clip_names = glob.glob(f"{audio_dir}/*.wav")
    audio_clip_names = \
        [cn.rsplit('.', maxsplit = 1)[0].rsplit('/', maxsplit = 1)[-1] for \
             cn in audio_clip_names]
    # keep only labels with associated audio
    clips_no_audio = []
    for idx in sentiment_labels.index:
        if idx not in audio_clip_names:
            clips_no_audio.append(idx)
    sentiment_labels.drop(index=clips_no_audio, inplace=True)
    drop_idx = sentiment_labels[sentiment_labels.sum(axis=1) > 1].index
    sentiment_labels.drop(index=drop_idx, inplace=True)
    print(f"{sentiment_labels.shape[0]} labels for sentiment with audio")
    return sentiment_labels

def main():
    labels_file = \
        Path(OUT_DIR / "sentiment_labels.csv")
    if Path.is_file(labels_file):
        print(labels_file, " already exists.")
        print("Use file to train")
    else:
        sentiment_labels = get_sentiment_labels(LABELS_DIR, AUDIO_CLIPS_DIR)
        sentiment_labels.to_csv(Path(OUT_DIR / "sentiment_labels.csv"),
            index  = True, header = True)
        print(f"labels exported to:\n {labels_file}")

if __name__ == "__main__":
    main()
