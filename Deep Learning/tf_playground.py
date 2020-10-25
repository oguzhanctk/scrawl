#%%
import tensorflow as tf

#placeholer is a promise that we give parameter this node later
w = tf.Variable([.5], tf.float32)
b = tf.Variable([.1], tf.float32)
X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#%%
linear_model = w * X + b
loss = tf.reduce_sum(tf.square(linear_model - y))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for _i in range(1000):
    sess.run([train], {X:[0, 1, 2, 3], y:[2, 4, 6, 9]})

print(sess.run([w, b]))
# %%
#perceptron presentation in tensorflow
import numpy as np
import tensorflow as tf

#%%
def step(x):
    return tf.sigmoid(x)

#%%
bias = 1.

train_in = [
    [1, 1, bias],
    [1, 0, bias],
    [0, 1, bias],
    [0, 0, bias]
]

train_out = [
    [1],
    [0],
    [0],
    [0]
]

w = tf.Variable(tf.random_normal([3, 1]))

output = step(tf.matmul(train_in, w))

error = tf.subtract(tf.cast(train_out, tf.float32), output)

mse = tf.reduce_mean(tf.square(error))

delta = tf.matmul(train_in, error, transpose_a=True)
train = tf.assign(w, tf.add(w, delta)) 

sess = tf.Session()
sess.run(tf.initialize_all_variables())

err, target = 1, 0
epoch, max_epoch = 0, 10
while err > target and epoch < max_epoch:
    epoch += 1
    err, _ = sess.run([mse, train])
    print(f"epoch : {epoch}\nmse : {err}")


# %%
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

sess = tf.InteractiveSession()

X = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y_ = tf.matmul(X, w) + b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

for _i in range(1000):
    batch = mnist.train.next_batch(100)
    if _i % 100 == 0:
        print(loss.eval(feed_dict={X : batch[0], y : batch[1]}))
    optimizer.run(feed_dict={X : batch[0], y : batch[1]})
    
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={X : mnist.test.images, y : mnist.test.labels}))

# %%
