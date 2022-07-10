import tensorflow as tf
import numpy as np
from transformers import BertTokenizerFast, TFBertModel


#Création du feature selon emotion positive ou négative (non-positive)
def pos_or_neg(row):
    if row['happiness'] or row['surprise']:
        return 1
    else :
        return 0

def tokenize(data, tokenizer,max_len=100) :
    input_ids = []
    attention_masks = []
    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)

def create_model(bert_model, max_len= 100):
    
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-7)
    loss = tf.keras.losses.binary_crossentropy

    input_ids = tf.keras.Input(shape=(max_len,),dtype='int32')
    
    attention_masks = tf.keras.Input(shape=(max_len,),dtype='int32')
    
    embeddings = bert_model([input_ids,attention_masks])[1]
    
    output = tf.keras.layers.Dense(1, activation="sigmoid")(embeddings)
    
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks], outputs = output)
    
    model.compile(opt, loss=loss, metrics=['binary_accuracy'])
    
    
    return model

def predict_emotion(sentence, model):

    encoded_str = tokenize(sentence)
    print(encoded_str)
    preds = model.predict(encoded_str)
    result_emotion = pos_or_neg(preds)

    return result_emotion