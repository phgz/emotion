# tools to split audio data

import glob
from datetime import timedelta
from time import time

import numpy as np
import pandas as pd


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
    df_labels['positive'] = \
        df_labels['sentiment'].map(lambda x : 1 if x > 0 else 0)
    df_labels['negative'] = \
        df_labels['sentiment'].map(lambda x : 1 if x < 0 else 0)

    # drop sentiment column (now in positive/negative)
    df_labels.drop(columns='sentiment', inplace=True)

    # if none of emotion or sentiment == 1, set none to 1
    df_labels['none'] = 0
    none_idx = \
        df_labels[df_labels[df_labels.columns[2:]].sum(axis=1) == 0].index

    df_labels.loc[none_idx,'none'] = 1

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


def get_labeled_clips(labels_dir):
    '''
        get id & clip number from labels
        labels_dir : directory containing raw labels
        returns   : dict of ids with correponding labeled clips
                    {'id' : [clip1, clip2, ... clip_n]}
    '''
    labeled_clips = {}
    labels = load_all_ratings(labels_dir, split_id_clip=True)

    labels = labels.drop_duplicates(subset=['id','clip'])

    uniq_ids = sorted(list(labels['id'].unique()))
    for i in uniq_ids:
        labeled_clips[i] = \
            sorted(
                labels[labels['id'] == i]['clip'].astype(int).to_list()
            )
    del labels
    return labeled_clips


def get_clips_info_from_text(text_dir, get_text=False):
    '''
        get audio information from text files to split audio into clips
        text_dir         : directory containing texts with audio info
        get_text         : add text column to info DataFrame
        returns: DataFrame
        - id (file id)
        - clip (clip number)
        - start_time (audio file segment start time)
        - end_time   (audio file segment end time)
        - len (audio clip len)
    '''
    text_files = glob.glob(f"{text_dir}/*.txt")
    dfs = []
    for file in text_files:
        with open(file, 'r') as text_file:
            lines = text_file.readlines()
            clips_info = []
            for line in lines:
                line_info = {}
                split_line = line.split('___')
                line_info['id'] = split_line[0]
                line_info['clip'] = int(split_line[1])
                line_info['start_time'] = round(abs(float(split_line[2])) ,3)
                line_info['end_time'] = round(float(split_line[3]), 3)
                line_info['len'] = \
                    line_info['end_time'] - line_info['start_time']

                if get_text:
                    line_info['text'] = split_line[4]
                
                clips_info.append(line_info)
        dfs.append(pd.DataFrame(clips_info))
    dfs = pd.concat(dfs, axis=0, ignore_index=True)
    return dfs


def get_labeled_clips_info(labels_dir, text_dir,
        get_text=False, show_progress=True):
    '''
        labels_dir       : directory containing labels
        text_dir         : directory containing text files and clip info
        get_text         : add text column to info DataFrame
        returns          : DataFrame containing:
                           id         : file id
                           clip       : audio clip/line number
                           start_time : audio clip start time
                           end time   : audio clip end time
                           len        : audio clip len
                           text       : text if get_text == True
                    clips_not_in_text (labeled_clips)
    '''

    stime = time()
    print("Recherche des ids et no. de clips annotés   ...")
    labeled_clips = get_labeled_clips(labels_dir)
    print("Extraction des infos audio (début/fin)      ...")
    clips_info = get_clips_info_from_text(text_dir, get_text)

    labeled_clips_info = []
    clips_not_in_text = []
    print("Selection de l'info audio des clips annotés ...")
    num_files = len(labeled_clips)
    files_processed = 0
    for i, clips in labeled_clips.items():
        mask_id = clips_info['id'] == i
        for clip in clips:
            mask_clip = clips_info['clip'] == clip
            clip_info = {}
            clip_info['id'] = i
            clip_info['clip'] = clip
            try:
                # if text with id contains clip, add info to clip_info
                len(clips_info[(mask_id) & (mask_clip)].index) == 1
                clip_info['start_time'] = \
                        clips_info[(mask_id) & (mask_clip)]['start_time'].values[0]
                clip_info['end_time'] = \
                        clips_info[(mask_id) & (mask_clip)]['end_time'].values[0]
                clip_info['len'] = \
                        clips_info[(mask_id) & (mask_clip)]['len'].values[0]
                if get_text:
                        clip_info['text'] = \
                            clips_info[(mask_id) & (mask_clip)]['text'].values[0]
                labeled_clips_info.append(clip_info)
            except:
                # if no clip in text add id and clip to errors
                clips_not_in_text.append(clip_info)

        if show_progress:
            files_processed += 1
            if files_processed % 10 == 0:
                print('.', end = '')
                if files_processed % 500 == 0:
                    print(f" {files_processed} de {num_files} fichiers")
    print("\nTemps d'exécution: ",
            f"{timedelta(seconds = round(time() - stime))} (h:mm:ss)")
    clips_not_in_text = pd.DataFrame(clips_not_in_text)
    labeled_clips_info = pd.DataFrame(labeled_clips_info)
    print(f"\n{labeled_clips_info.shape[0]} clips annotés")
    print(clips_not_in_text.shape[0],
        "clips annotés sans info audio")
    return labeled_clips_info, clips_not_in_text
