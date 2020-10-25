#%%
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization, Dropout  

#%%
#import data
mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#%%
#prepare image data for model 
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float") / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float") / 255

# %%
#create model
model = keras.Sequential([
    keras.layers.Conv2D(64, (7, 7), padding="same", activation="relu", input_shape=(28,28,1)),
    BatchNormalization(),
    Dropout(0.25),

    keras.layers.MaxPooling2D((2, 2)),
    BatchNormalization(),
    Dropout(0.25),

    keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    Dropout(0.25),

    keras.layers.MaxPooling2D((2, 2)),
    BatchNormalization(),
    Dropout(0.25),

    keras.layers.Flatten(),
    
    keras.layers.Dense(10, activation="softmax")
    
    ])



model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_images, train_labels, batch_size=250, epochs=5, verbose=1)
model.evaluate(test_images, test_labels)
# %%



