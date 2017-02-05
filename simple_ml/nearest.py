import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
"""
Implement the nearest image algorithm
"""

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
Xtrain, Ytrain = mnist.train.next_batch(5000)
Xtest, Ytest = mnist.test.next_batch(200)

# process the test data to numpy.1darray (list of class type)
Ytrain = np.where(Ytrain == 1)[1]
Ytest = np.where(Ytest == 1)[1]

x_train = tf.placeholder(tf.float32, shape = [None, 784], name = 'x_train')
y_train = tf.placeholder(tf.int16, shape = [None], name = 'y_train')
x_test = tf.placeholder(tf.float32, shape = [None, 784], name = 'x_test')
y_test = tf.placeholder(tf.int16, shape = [None], name = 'y_test')

x_train_shape  = tf.shape(x_train)
x_test_shape = tf.shape(x_test)
x_test_tiled = tf.reshape(tf.tile(x_test, [1, x_train_shape[0]]), 
        [x_test_shape[0], x_train_shape[0], 784]) 

# the distance of each train set to each test set (axis 0 is test, axis 1 is train)
distance = tf.reduce_sum(tf.abs(x_test_tiled - x_train), 2)
best_fit = tf.argmin(distance, 1)
# tensor indexing to find the test set result (this is result, the rest is testing accuracy)
nearest = tf.gather(y_train, best_fit)
# how many among the test results are correct
correctness = tf.reduce_sum(tf.cast(tf.equal(nearest, y_test), tf.float32), 0) 
accuracy = correctness / tf.cast(x_test_shape[0], tf.float32)

# initialize the variables
init = tf.initialize_all_variables()

feed = {x_train: Xtrain, y_train: Ytrain, x_test: Xtest, y_test: Ytest}
sess = tf.Session()
sess.run(init)
print "accuracy", sess.run(accuracy, feed)








