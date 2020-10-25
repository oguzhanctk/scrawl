#%%
def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def error(target, predicted):
    return np.sum((target - predicted) ** 2) / 2

def show_weights(weights, step):
    for _i in range(0, len(weights) , step):
        print(f"in {_i} -->:\n {weights[_i]}")
    print(f"in {len(weights)} -->:\n {weights.pop()}")
#%%
import numpy as np
import matplotlib.pyplot as plt

# %%
_LOOP = 300
inputs = np.array([[0, 0, 1, 1], [1, 1, 1, 0], [1, 0, 1, 0], [1, 1, 1, 0], [0, 1, 0, 1], [0, 0, 0, 1]])
target = np.array([[0, 1, 1, 0, 0, 0]]).T
weights = np.random.random((1, len(inputs[0]))).T
loss = []
weights_list = []

for _i in range(_LOOP):
    z = np.dot(inputs, weights)
    predictions = sigmoid(z)
    loss.append(error(target, predictions))
    weights = weights - 0.1 * np.dot(inputs.T, (predictions - target) * (predictions * (1 - predictions)))
    weights_list.append(weights)

show_weights(weights_list, 100)

#visualization of Loss value

plt.plot(range(_LOOP), loss)    
plt.ylabel('Loss')
plt.show()


#%%
#prediction
"""
x1 feature has perfect correlation with target values 
and now prediction correlates with x1 variable(x1 = 1 <-> 0, prediction ~ 1 <-> 0 )
"""

test = np.array([[1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 1, 1]]) 
s = sigmoid(np.dot(test, weights))
print(s)



# %%

a = np.array([[1, 4, 5, 31],
              [5, 2, 1, 1]])

b = np.array([[4.1, -2.1, 5.1, 2.4],
              [2.1, 5.2, -0.2, 0.2]])

# np.random.seed(23)
print(np.dot(a, b.T))
print(np.zeros((1, 3)))
# %%


# %%
