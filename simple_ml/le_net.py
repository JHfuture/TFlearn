import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as read_data

"""
Implement LeNet in the model
conv, relu, maxpool, conv, relu, maxpool, fc, relu, dropout, softmax
loss is calculated by cross-entropy
"""

mnist = read_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.01
batch_size = 100
validation_size = 1000
episode_count = 100

def weight_variable(shape):
    # output a weight variable with the shape in the argument, shape is a tuple
    # initialize by Xavier initialization
    # the last variable is output size, the first few multiply will be input size
    Nin = np.prod(shape) / shape[-1]
    Nout = shape[-1]
    variance = np.sqrt(3 / (Nin + Nout))
    w_init = np.random.normal(scale = variance, size = shape)
    w = tf.Variable(w_init, dtype = tf.float32, name = 'weight')
    return w

def bias_variable(shape):
    # output a bias variable with the shape in the argument, shape is a tuple
    b_init = np.zeros(shape)
    b = tf.Variable(b_init, dtype = tf.float32, name = 'bias')
    return b

def conv_relu(input_data, w, b):
    # implement conv2d, then add bias, and put on relu
    conv = tf.nn.conv2d(input_data, w, [1,1,1,1], "SAME") + b
    h = tf.nn.relu(conv)
    return h

def fc_relu(input_data, w, b):
    # implement fully connected, then put on relu
    h = tf.nn.relu(tf.matmul(input_data, w) + b)
    return h

def max_pool(input_data):
    # implement max pooling 2*2
    return tf.nn.max_pool(input_data, [1,2,2,1], [1,2,2,1], "SAME")

def drop_out(input_data, keep_prob):
    # dropout by the probability of keep_prob from the input data
    return tf.nn.dropout(input_data, keep_prob)

x_train = tf.placeholder(tf.float32, [None, 784])
y_train = tf.placeholder(tf.bool, [None, 10])


x = tf.reshape(x_train, shape = [-1,28,28,1])
with tf.name_scope('conv_layer1'):
    w1 = weight_variable((5,5,1,32))
    b1 = bias_variable((32))

    h1 = conv_relu(x, w1, b1)
    h1_maxpool = max_pool(h1)
with tf.name_scope('conv_layer2'):
    w2 = weight_variable((5,5,32,64))
    b2 = bias_variable((64))

    h2 = conv_relu(h1_maxpool, w2, b2)
    h2_maxpool = max_pool(h2)
with tf.name_scope('fc_layer1'):
    w3 = weight_variable((7*7*64, 1024))
    b3 = bias_variable((1024))

    h2_reshape = tf.reshape(h2_maxpool, [-1, 7*7*64])
    h3 = fc_relu(h2_reshape, w3, b3)
    keep_prob = tf.placeholder(tf.float32)
    h3_drop = drop_out(h3, keep_prob)
with tf.name_scope('readout_layer'):
    w4 = weight_variable((1024, 10))
    b4 = bias_variable((10))

    y_conv = tf.matmul(h3_drop, w4) + b4

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, tf.cast(y_train, tf.float32)), 0)

optimizer = tf.train.AdamOptimizer(learning_rate)
adam = optimizer.minimize(loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.where(y_train)[:, 1], tf.argmax(y_conv, 1)), tf.float32), 0)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for episode_index in range(episode_count):
    Xtrain, Ytrain = mnist.train.next_batch(batch_size)
    kp = 0.5
    feed = {x_train: Xtrain, y_train: Ytrain, keep_prob: kp}

    _, l = sess.run([adam, loss], feed)
    
    Xval, Yval = mnist.validation.next_batch(validation_size)
    feed = {x_train: Xval, y_train: Yval, keep_prob: kp}
    a = sess.run(accuracy, feed)
    print "episode: " + str(episode_index + 1) + " ,loss: ", l, " ,accuracy: ",a




