import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

x_data = np.random.uniform(0,1, size = 100).astype(np.float32)
noise = np.random.normal(scale = 0.01, size = len(x_data))
y_data = 0.1 * x_data + 0.3 + noise

plt.plot(x_data, y_data, 'ro')

x_holder = tf.placeholder(tf.float32, shape = (100), name = 'x_holder')
y_holder = tf.placeholder(tf.float32, shape = (100), name = 'y_holder')

W = tf.Variable(random.gauss(0, 0.01), name = 'weight')
b = tf.Variable(0., name = 'bias')
y = W * x_holder + tf.ones([100]) * b

loss = tf.reduce_sum(tf.square(y - y_data), 0)
optimizer = tf.train.AdamOptimizer(0.01)
train_op = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
feed_dict = {x_holder: x_data, y_holder: y_data}
for _ in range(500):
    _, loss_value = sess.run([train_op, loss], feed_dict)
    weight, bias = sess.run([W,b])
    print "w: ", weight, "b: ", bias
    print "loss: ", loss_value
    print ""

