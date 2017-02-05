import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

hello = tf.constant("Hello TensorFlow!")
with tf.Session() as sess:
    print sess.run(hello)

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

mul = tf.mul(a,b)
with tf.Session() as sess:
    print sess.run(mul, feed_dict = {a:2, b:3})

a = np.arange(0,6).reshape(2,3)
b = np.arange(0,9).reshape(3,3)
a = tf.constant(a.tolist())
b = tf.constant(b.tolist())

product = tf.matmul(a,b)
with tf.Session() as sess:
    result = sess.run(product)
    print result
    print type(result)
