# split audio files (.wav) into clips coressponding to text

# import os
import pathlib
import subprocess
from datetime import timedelta
from glob import glob
from pathlib import Path
from time import sleep, time

import pandas as pd
from audio_utils import get_labeled_clips_info

from emotion import root_dir

FFMPEG = "/usr/bin/ffmpeg"
LABELS_DIR = Path(root_dir / "data/raw/labels")
TEXT_DIR = Path(root_dir / "data/raw/text")
AUDIO_RAW_DIR = Path(root_dir / "data/raw/audio")
AUDIO_CLIPS_DIR = Path(root_dir / "data/interim/audio")

def split_audio_clip(clip_info, audio_raw_dir, audio_out_dir):
    '''
        get part of audio file (.wav) from start_time to end_time
        clip_info    : pd.Series containing id, clip, start_time, end_time
                        (info on wav file section to get)
        audio_raw_dir : directory containing full wav files (to be splited)
        audio_out_dir : directory where split clips will be saved
    '''

    if subprocess.run([FFMPEG, "-i",
                       Path(audio_raw_dir / f"{clip_info['id']}.wav"),
                       "-ss", str(clip_info['start_time']),
                       "-to", str(clip_info['end_time']),
                       Path(audio_out_dir / \
                            f"{clip_info['id']}_{str(clip_info['clip'])}.wav"),
                       "-hide_banner"]).returncode != 0:
        clip_err = {"id" : clip_info['id'], "clip" : clip_info['id']}
        print(clip_err)
        return clip_err
    else:
        return 0


def split_audio_clips(audio_raw_dir, audio_out_dir, clips_info=None):
    '''
        split full audio files to clips corresponding to text comments
        audio_raw_dir : directory containing full wav files (to be splited)
        audio_out_dir : directory where split clips will be saved
        clips_info    : pd.DataFrame containing id, clip, start_time, end_time
                        (info to split wav files)
        returns       : list containing files where errors occcured
    '''

    out_path = pathlib.Path(audio_out_dir)
    out_path.mkdir(parents=True, exist_ok =True)

    if len(glob(f"{out_path}/*.wav")) > 0:
        print(f"\naudio_out_dir: ({audio_out_dir})",
                "not empty, exiting")
        return
    else:
        print("No wavs in directory, ok")

    if clips_info is None:
        clips_info, _  = \
                get_labeled_clips_info(LABELS_DIR, TEXT_DIR, get_text = False)
    elif isinstance(clips_info, pd.DataFrame):
        info_columns = clips_info.columns.tolist()
        for c  in ['id', 'clip', 'start_time', 'end_time']:
            if c not in info_columns:
                print("DataFrame columns must be: ",
                    "['id', 'clip', 'start_time', 'end_time']")
                return
    else:        
        print("\nclips_info must be pandas DataFrame")
        print("containing columns: id, clip, start_time, end_time")
        return

    split_errors = []
    num_split_clips = 0
    print("Séparaton des fichiers audio en clips ...")
    sleep(1)
    stime = time()
    for i in clips_info.index:
        curr_clip = clips_info.loc[i]
        split_error = split_audio_clip(curr_clip, audio_raw_dir, audio_out_dir)
        if split_error != 0:
            split_errors.append(split_error)
        else:
            num_split_clips += 1
    print(f"\nSéparation des fichiers audio terminée, {num_split_clips} clips")
    print(f"\nTemps d'exécution: ",
        f"{timedelta(seconds = round(time() - stime))} (h:mm:ss)")
    return split_errors

def main():
    clips_info, _ = get_labeled_clips_info(LABELS_DIR, TEXT_DIR,
                            get_text = False)
        
    split_errors = \
            split_audio_clips(AUDIO_RAW_DIR, AUDIO_CLIPS_DIR, clips_info)
    print(f"\n{len(split_errors)} split errors:")
    print(split_errors)

if __name__ == "__main__":
    main()
