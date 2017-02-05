import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
"""
implement multiple fully connected layer neural network
every layer is followed by a relu layer
with a softmax function at the end
loss is set as cross-entropy
"""

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 100
hidden_layer_count = 2
layer_size_list = [500, 50]
learning_rate = 0.001
validation_size = 1000
episode_count = 750
assert len(layer_size_list) == hidden_layer_count, "layer count and layer size length not match"

x_train = tf.placeholder(tf.float32, [None, 784])
y_train = tf.placeholder(tf.float32, [None, 10])
x_val = tf.placeholder(tf.float32, [None, 784])
y_val = tf.placeholder(tf.bool, [None, 10])
x_test = tf.placeholder(tf.float32, [None, 784])
y_test = tf.placeholder(tf.bool, [None, 10])

nn_param = {}

input_data = x_train
val_data = x_val
input_size = 784
for layer_index in range(hidden_layer_count):
    w_name = "W" + str(layer_index + 1)
    b_name = "b" + str(layer_index + 1)

    layer_size = layer_size_list[layer_index]
    init_w = np.random.random((input_size, layer_size)) / np.sqrt(input_size / 2.)
    init_b = np.zeros(layer_size)

    w = tf.Variable(init_w, dtype = tf.float32, name = w_name)
    b = tf.Variable(init_b, dtype = tf.float32, name = b_name)
    
    nn_param[w_name] = w
    nn_param[b_name] = b

    input_data = tf.nn.relu(tf.matmul(input_data, w) + b)
    val_data = tf.nn.relu(tf.matmul(val_data, w) + b)
    input_size = layer_size

# last layer, which is not hidden
w_name = "W" + str(hidden_layer_count + 1)
b_name = "b" + str(hidden_layer_count + 1)

init_w = np.random.random((input_size, 10)) / np.sqrt(input_size / 2.)
init_b = np.zeros(10)
w = tf.Variable(init_w, dtype = tf.float32, name = w_name)
b = tf.Variable(init_b, dtype = tf.float32, name = b_name)
nn_param[w_name] = w
nn_param[b_name] = b

prediction = tf.nn.softmax(tf.matmul(input_data, w) + b) + 1e-8  # 1e-8 is for prevent loss to nan
loss = tf.reduce_mean( - tf.reduce_sum(y_train * tf.log(prediction), 1),0)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate)
adam = optimizer.minimize(loss)

prediction_val = tf.argmax(tf.matmul(val_data, w) + b, 1)
correctness_val = tf.cast(tf.equal(tf.where(y_val)[:,1], prediction_val), tf.float32)
accuracy_val = tf.reduce_mean(correctness_val, 0)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for episode_index in range(episode_count):
    Xtrain, Ytrain = mnist.train.next_batch(batch_size)
    Xval, Yval = mnist.validation.next_batch(validation_size)
    feed = {x_train: Xtrain, y_train: Ytrain, x_val: Xval, y_val: Yval}

    _, l, a = sess.run([adam, loss, accuracy_val], feed)
    print "episode number: ", episode_index + 1, " ,loss: ", l, " ,accuracy: ", a

print "========= test ============"
Xtest = mnist.test.images
Ytest = mnist.test.labels
feed = {x_val: Xtest, y_val:Ytest}
a = sess.run(accuracy_val, feed)
print "test accuracy: ", a






