#%%
import tensorflow as tf
import pandas as pd
from tensorflow import keras

import nltk
from nltk import word_tokenize
nltk.download("stopwords")
from nltk.corpus import stopwords 
stop = set(stopwords.words("english"))

from nltk.stem import WordNetLemmatizer as wnl
import re

#%%

def clean_text_data(data):
    regex = re.compile("[^a-zA-Z]")
    stem_data_without_stop = [" ".join([wnl().lemmatize(regex.sub("", word.lower())) for word in text.split(" ") if word not in stop]) for text in data]
    return stem_data_without_stop

#%%
raw_data = pd.read_csv("gender-classifier-DFE-791531.csv", encoding="latin")
raw_data.dropna(subset=["text", "gender"], inplace=True)
train_data = raw_data.loc[:12000, ["text"]]
train_label = raw_data.loc[:12000, ["gender"]]
test_data = raw_data.loc[12000:, ["text"]]
test_label = raw_data.loc[12000:, ["gender"]]
#%%
train_data["text"] = clean_text_data(train_data["text"].values)
test_data["text"] = clean_text_data(test_data["text"].values)

train_label = [0 if label == "male" 
    else 1 if label == "female" 
    else 2 if label == "brand" 
    else 3 for label in train_label.gender]


test_label = [0 if label == "male" 
    else 1 if label == "female" 
    else 2 if label == "brand" 
    else 3 for label in test_label.gender]

# %%
MAX_NUM_WORDS = 10000

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=MAX_NUM_WORDS,
    oov_token="<UNK>",
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, 
    split=' ')

tokenizer.fit_on_texts(train_data.text.values)  


train_sequences = tokenizer.texts_to_sequences(train_data.text.values) 
test_sequences = tokenizer.texts_to_sequences(test_data.text.values)

train_padded = keras.preprocessing.sequence.pad_sequences(train_sequences, padding="post", truncating="post", maxlen=140)
test_padded = keras.preprocessing.sequence.pad_sequences(test_sequences, padding="post", truncating="post", maxlen=140)

# %%
#create model

model = keras.Sequential()

model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(4, activation="softmax"))

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# %%
#split train and validation data
train_X = train_padded[:9000] 
train_y = train_label[:9000]

val_X = train_padded[9000:] 
val_y = train_label[9000:]

#%%
model.fit(train_X, train_y, epochs=30, batch_size=512, validation_data=(val_X, val_y), verbose=1)

model.evaluate(test_padded, test_label)

# %%
