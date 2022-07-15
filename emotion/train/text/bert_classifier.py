
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text
import tensorflow_hub as hub


from pathlib import Path
from emotion import module_dir, root_dir
from tensorflow.keras.models import model_from_json
from transformers import BertTokenizerFast, TFBertModel
from sklearn.model_selection import train_test_split as tts

from emotion import module_dir, root_dir

MAX_LEN = 20


module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
BERT_LAYER = hub.KerasLayer(module_url, trainable=True)
TOKENIZER = BertTokenizerFast.from_pretrained('bert-base-uncased')
DATA_DIR = Path(root_dir / 'data/processed/text')
FEATURES = Path(DATA_DIR / 'polarity.csv')

ARTIFACTS_DIR = Path(module_dir / 'artifacts')


#Interesting/meaningful metrics
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name = 'accuracy'),
    tf.keras.metrics.Precision(name = 'precision'),
    tf.keras.metrics.Recall(name = 'recall')
]


#Encodes lists of texts into BERT-useable tensors
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


#Building a neural-net that uses BERT embeddings
def build_model(bert_layer=BERT_LAYER, max_len=MAX_LEN):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]

    #net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    #net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(3, activation='softmax')(net)
    
    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=METRICS)
    
    return model

#Simple training function with a few parameters
def train_dnn(model, X_train, y_train, e=4):
    checkpoint = tf.keras.callbacks.ModelCheckpoint('checkpoint_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

    train_input = bert_encode(X_train)
    train_labels = y_train

    train_history = model.fit(
            train_input, train_labels, 
            validation_split=0.2,
            epochs=e,
            callbacks=[checkpoint, earlystopping],
            batch_size=4,
            verbose=1
            )
    return train_history


def main():
    if Path.is_file(FEATURES):
        data = pd.read_csv(FEATURES)
        x = data.clean_text.values
        dummy_sents = pd.get_dummies(data.sentiment)
        y = dummy_sents.values
        X_train, X_test, y_train, y_test = tts(x, y, test_size = 0.1)
        model = build_model()

        train_dnn(model, X_train, y_train)

        model_json = model.to_json()
        with open(f"{ARTIFACTS_DIR}/text_model.json", "w") as json_file:
            json_file.write(model_json)
        
        model.save_weights(f"{ARTIFACTS_DIR}/weights.h5")

    else:
        print(f"MISSING FILES FROM {DATA_DIR}")



if __name__ == "__main__":
    main()