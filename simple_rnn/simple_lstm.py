import tensorflow as tf
#from tensorflow.contrib import rnn
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

"""
implement simple lstm in mnist
"""

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

memory_unit = 128
learning_rate = 0.001
batch_size = 50
epoch_count = 2000
validation_size = 100

step_size = 28
input_size = 28
class_size = 10

x = tf.placeholder(tf.float32, [None, step_size, input_size])
y = tf.placeholder(tf.float32, [None, class_size])

def RNN(data, w, b):
    # data is in the type [steps, batch_size, input_size]
    lstm = tf.nn.rnn_cell.BasicLSTMCell(memory_unit, state_is_tuple = True)
    output, state = tf.nn.rnn(lstm, data, dtype = tf.float32)
    y_rnn = tf.matmul(output[-1], w) + b
    return y_rnn

w = tf.Variable(tf.random_normal([memory_unit, class_size]), 'weight')
b = tf.Variable(tf.random_normal([class_size]), 'bias')

data = tf.transpose(x, [1,0,2])
data= tf.reshape(data, [-1, input_size])
data = tf.split(0, step_size, data)

y_rnn = RNN(data, w, b)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_rnn, y), 0)

optimizer = tf.train.AdamOptimizer(learning_rate)
adam = optimizer.minimize(loss)

correctness = tf.where(tf.cast(y, tf.bool))[:, 1]
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_rnn, 1), correctness), tf.float32), 0)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for epoch_index in range(epoch_count):
    Xtrain, Ytrain = mnist.train.next_batch(batch_size)
    Xtrain = np.reshape(Xtrain, (batch_size, step_size, input_size))
    feed = {x: Xtrain, y: Ytrain}

    _, l = sess.run([adam, loss], feed)

    Xval, Yval = mnist.validation.next_batch(validation_size)
    Xval = np.reshape(Xval, (validation_size, step_size, input_size))
    feed = {x: Xval, y: Yval}

    a = sess.run(accuracy, feed)

    print "episode: ", (epoch_index + 1), " ,loss: ", l, " ,accuracy: ", a

Xtest = mnist.test.images
Ytest = mnist.test.labels
Xtest = np.reshape(Xtest, (-1, step_size, input_size))
feed = {x:Xtest, y:Ytest}
print "========== test case ==========="
print "test accuracy: ", sess.run(accuracy, feed)

