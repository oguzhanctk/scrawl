#%%
import tensorflow as tf
from tensorflow import keras
import numpy as np

#%%
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

#%%

data = keras.datasets.imdb

(train_data, train_label), (test_data, test_label) = data.load_data(num_words=10000)

# %%
word_index = data.get_word_index()

word_index = {k : v + 3 for k, v in word_index.items()}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

# %%
#create model
model = keras.Sequential()

model.add(keras.layers.Embedding(10000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# %%
#split validation and train set
X_val = train_data[:10000] 
y_val = train_label[:10000]

X_train = train_data[10000:]
y_train = train_label[10000:]

# %%
#fit model
fit_model = model.fit(X_train, y_train, epochs=30, batch_size=512, validation_data=(X_val, y_val), verbose=1)

#%%
#save model
model.save("model.h5")

#model loading
# model_2 = keras.models.load_model("model.h5")
# model_2.summary()
#%%
#evaluate
results = model.evaluate(test_data, test_label)

#%%
#predict
c = 0
preds = model.predict(test_data[:10])
for _i in range(10):
    print(f"actual value : {test_label[_i]}\nprediction : {preds[_i]}")
    print("--------------------------")

# %%



# %%


# %%
