#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

# %%
#define a model
model = keras.Sequential() #create a sequential model

model.add(keras.layers.Conv2D(32, (5, 5), activation="relu", input_shape=(28, 28, 1))) #use filter with 32 channel
model.add(keras.layers.MaxPooling2D((2, 2))) #pooling for reduce dimensionality

model.add(keras.layers.Conv2D(64, (5, 5), activation="relu"))
model.add(keras.layers.MaxPooling2D(2, 2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation="softmax"))

#%%

mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_img = test_images

#%%

#prepare train and test image for model
train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images.astype("float") / 255

test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images.astype("float") / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# %%

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.fit(train_images, train_labels, batch_size=100, epochs=5, verbose=1)
test_loss, test_acc = model.evaluate(test_images, test_labels)

#%%

print(f"Test accuracy : {test_acc}")

# %%

def compare(data, pred, n):
    for _i in range(n):
        # if np.argmax(pred[_i]) != data[_i]:
        #     break
        print(_i)
        print(f"prediction-{_i}: {np.argmax(pred[_i])}\nreal : {data[_i]}")


# %%
pred = model.predict(test_images[0:34])
data = np.argmax(test_labels[0:34], axis=1)
compare(data, pred, 34)

# %%
