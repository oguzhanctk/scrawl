#%%
#MNIST dataset example
import tflearn.datasets.mnist as mnist
import tensorflow as tf
import tflearn

#%%
X, Y, testX, testY = mnist.load_data(one_hot=True)

#%%

input_layer = tflearn.input_data(shape=[None, 784])

hidden_layer1 = tflearn.fully_connected(input_layer, 256, activation="relu", regularizer="L2", weight_decay=0.001)
dropout1 = tflearn.dropout(hidden_layer1, 0.8) #0.6 - 0.8 optimal

hidden_layer2 = tflearn.fully_connected(dropout1, 256, activation="relu", regularizer="L2", weight_decay=0.001)
dropout2 = tflearn.dropout(hidden_layer2, 0.8)

softmax = tflearn.fully_connected(dropout2, 10, activation="softmax")

sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=100)
top_k = tflearn.metrics.Top_k(3)
net = tflearn.regression(softmax, optimizer=sgd, metric=top_k, loss="categorical_crossentropy")

model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=10, validation_set=(testX, testY), show_metric=True, run_id="dense_model")


# %%
from PIL import Image
import numpy as np
import scipy

image = Image.open("test.png").convert("L")
data = np.asarray(image).flatten()
x = np.asarray(list(map(lambda x : (255 - x) / 255, data)))
pred = model.predict(x.reshape(1, -1))
print(pred)

# %%

# %%
