import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
"""
implement one layer neural network (no hidden layer)
with a softmax function at the end
"""

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

batch_size = 1000
training_epoch = 250
learning_rate = 0.01
validation_size = 1000

x_train = tf.placeholder(tf.float32, [None, 784], name = "x_train")
y_train = tf.placeholder(tf.float32, [None, 10], name = "y_train")
x_validation = tf.placeholder(tf.float32, [None, 784], name = "x_validation")
y_validation = tf.placeholder(tf.bool, [None, 10], name = "y_validation")

W = tf.Variable(np.random.random((784,10)) * 0.01, dtype = tf.float32, name = "weight")
b = tf.Variable(tf.zeros([10]), dtype = tf.float32, name = "bias")

prediction = tf.nn.softmax(tf.matmul(x_train, W) + b)
loss =  tf.reduce_mean(- tf.reduce_sum(y_train * tf.log(prediction), 1))

optimizer = tf.train.AdamOptimizer(learning_rate)
adam = optimizer.minimize(loss)


fact_val = tf.where(y_validation)[:,1]
prediction_val = tf.argmax(tf.matmul(x_validation, W) + b, 1)
accuracy = tf.reduce_sum(tf.cast(tf.equal(fact_val, prediction_val), tf.float32), 0) / tf.cast(validation_size, tf.float32)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for epoch_index in range(training_epoch):
    Xtrain, Ytrain = mnist.train.next_batch(batch_size)
    Xvalidation, Yvalidation = mnist.validation.next_batch(validation_size)

    feed = {x_train: Xtrain, y_train : Ytrain, x_validation : Xvalidation, y_validation : Yvalidation}

    _, loss_val, accuracy_val = sess.run([adam, loss, accuracy], feed)
    print "epoch number: ", epoch_index + 1, " ,loss: ", loss_val, " ,accuracy: ", accuracy_val

