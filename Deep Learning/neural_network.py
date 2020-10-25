#%%
import numpy as np

#%%
np.random.seed(23)
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.0]]

#%%
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases

class Relu_Activation:
    def forward(self, inputs):
        self.outputs = np.maximum(0, inputs)    

relu_activation = Relu_Activation()

layer_1 = Layer_Dense(4, 5)
layer_1.forward(X)
relu_activation.forward(layer_1.outputs)
print(f"layer_1 :\n {relu_activation.outputs}")

layer_2 = Layer_Dense(5, 2)
layer_2.forward(layer_1.outputs)
relu_activation.forward(layer_1.outputs)
print(f"layer_2 :\n {relu_activation.outputs}")

layer_3 = Layer_Dense(2, 1)
layer_3.forward(layer_2.outputs)
relu_activation.forward(layer_3.outputs)
print(f"layer_3 :\n {relu_activation.outputs}")

 