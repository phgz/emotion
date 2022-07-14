

import re
import sys
import string
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter
from transformers import BertTokenizerFast, TFBertModel

from sklearn.model_selection import train_test_split as tts
from emotion import module_dir, root_dir



TOKENIZER = BertTokenizerFast.from_pretrained('bert-base-uncased')
ARTIFACTS_DIR = Path(module_dir / "artifacts")
DATA_DIR = Path(root_dir / 'data/processed/text')
DATA = f"{DATA_DIR}/polarity_df"

MAX_LEN = 20


#Removes time stamps from every line
def remove_stamps_str(line)->str:
    stamp = re.search('.+___', line).group(0)
    new_line = line.strip(stamp)
    return new_line

## Pour python < 3.9, sinon str.removeprefix() de base
def removeprefix(self: str, prefix: str, /) -> str:
    if self.startswith(prefix):
        return self[len(prefix):]
    else:
        return self[:]

#Retire chaque stamps, chaque texte devien 1 seul str
def text_list_generator(files_list, text_dir):
    text_list = []
    for filename in files_list:
        with open(file = filename, encoding = 'utf-8') as f:

            ##WINDOWS SPECIFIC
            if sys.platform == 'win32':
                videoid = removeprefix(filename, text_dir + '\\').rstrip('.txt')
            else :
                videoid = removeprefix(filename, text_dir + '/').rstrip('.txt')
            lines = f.readlines()
            for line_number, text_line in enumerate(lines):
                clean_line = remove_stamps_str(text_line)
                clip_id = videoid +'_'+ text_line.split('___')[1]
                #clip_id = videoid +'_' +str(line_number)
                yield (clip_id, clean_line.rstrip())

#Retire tous les timestamps en début de ligne, présents dans chaque transcript
def remove_stamps_str(line)->str:
    stamp = re.search('.+___', line).group(0)
    new_line = line.strip(stamp)
    return new_line

#Retire les charactères non-ascii 
def remove_nonascii(line)->str:
    ascii_line = line.encode(encoding = 'ascii', errors = 'ignore').decode()
    return ascii_line

#met tout en minuscules, retire les nombres
def clean_punct_digits(line)->str:
    new_line = line.translate(str.maketrans('', '', string.punctuation))
    new_line = ' '.join([word.lower() for word in new_line.split() if (len(word) >=2 and word.isalpha())])
    return new_line


def extract_text_from_dir(files_list, text_dir):
    corpus = (text for (identifier,text) in text_list_generator(files_list, text_dir))


#Uses the predefined bert Tokenizer to tokenize text segments
def bert_encode(texts, tokenizer=TOKENIZER, max_len=MAX_LEN):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def main():

    polarity_df = pd.read_csv(DATA)
    dummy_sents = pd.get_dummies(polarity_df['sentiment'])
    x = polarity_df.clean_text.values
    y = dummy_sents.values
    X_train, X_test, y_train, y_test = tts(x, y, test_size = 0.2)


if __name__ == "__main__":
    main()