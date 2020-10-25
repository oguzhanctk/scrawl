#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# %%
#define functions
def compare_preds(labels, preds):
    length = len(labels)
    err = 0
    for _i in range(length):
        if labels[_i] != np.argmax(preds[_i]):
            err += 1
    print(f"error rate : {(err + 1) / length}")

def compare_n_with_images(actual_X, actual_y, preds, n, is_labelled, labels=None):
    for _i in range(n):
        plt.imshow(actual_X[_i], cmap=plt.cm.binary)
        plt.xlabel(f"actual value : {actual_y[_i] if is_labelled == True else labels[actual_y[_i]]}")
        plt.title(f"Predicted value : {np.argmax(preds[_i]) if is_labelled else labels[np.argmax(preds[_i])]}")
        plt.show()
    
#%%
#import fashion data
data = keras.datasets.fashion_mnist
data_2 = keras.datasets.mnist

#split data to train and test
(train_X, train_y), (test_X, test_y) = data.load_data()
(train_numbers, train_labels), (test_numbers, test_labels) = data_2.load_data()

#normalize data
train_X = train_X / 255
test_X = test_X / 255
train_numbers = train_numbers / 255
test_numbers = test_numbers / 255

#define output labels for fashion data
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#%%
#FASHION DATA
#define model(inputs, layers, layer_dense(neurons), loss_functin, optimizer, metric) for fashion data
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)), #input layer
    keras.layers.Dense(128, activation="relu"), #dense means fully connected layer
    keras.layers.Dense(10, activation="softmax") #outputs
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_X, train_y, epochs=5)

# %%
#evaluate model loss and accuracy for test data
loss, acc = model.evaluate(test_X, test_y)
print(f"loss value : {loss}\n"
      f"accuracy value : {acc}")


# %%
#prediction and compare with actual data
preds = model.predict(test_X)
compare_preds(test_y, preds)
compare_n_with_images(test_X, test_y, preds, 5, False, labels)

#%%
#NUMBER DATA
#define model(inputs, layers, layer_dense(neurons), loss_function, optimizer, metrics) for number data
model_2 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    # keras.layers.Dropout(0.2),# to prevent overfitting(randomly ignore 0.2 of neurons in layer)
    keras.layers.Dense(10, activation="softmax")
])
model_2.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model_2.fit(train_numbers, train_labels, epochs=5)
model_2.evaluate(test_numbers, test_labels)

# %%
#predictions and compare with original data
preds_2 = model_2.predict(test_numbers)
compare_preds(test_labels, preds_2)
compare_n_with_images(test_numbers, test_labels, preds_2, 10, is_labelled=True)

# %%
#test model for ./test.png
from PIL import Image

image = Image.open("test.png").convert("L")
input_data = np.asarray([[[(255 - piksel) / 255 for piksel in row] for row in np.asarray(image)]])

p = model_2.predict(input_data)
print(np.argmax(p))




# %%

# %%
