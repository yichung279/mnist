from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

'''
sess = tf.InteractiveSession()


# 28px * 28px = 784
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b= tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss function (cross entropy) and learning rate
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

#argmax return index correct_prediction=[t, f, t, .........]
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.initialize_all_variables().run()
for i in range(10000):
    # random 100 data from all data (cheaper)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

print (accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))
'''

x = tf.placeholder(tf.float32, (None, 784))
y = tf.placeholder(tf.float32, (None, 10))

#! normal distribution v.s. zeros ?
# prevent 0 gradients and avoid "dead neurons"
w1 = tf.Variable(tf.truncated_normal(shape = (784, 200), stddev = 0.1))
b1 = tf.Variable(tf.constant([0.1] * 200))

#! hidden layer is important ?
w2 = tf.Variable(tf.truncated_normal(shape = (200, 10), stddev = 0.1))
b2 = tf.Variable(tf.constant([0.1] * 10))

h1 = tf.nn.relu(x @ w1 + b1)
output = h1 @ w2 + b2

loss = tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = y)

#! Adam v.s. SGD ?
# Different algorithm.SGD has right direction but slow.
#! learning rate 0.01 v.s. 0.1 ?
# too over = =
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):

    #! batch size 100 v.s. 256 ? 1 v.s. 10000 ?
    # 1 over fitting 10000 expensive
    batch_xs, batch_ys = mnist.train.next_batch(256)

    sess.run(train_op, feed_dict = {x: batch_xs, y: batch_ys})

    if i % 100 == 0:
        print(sess.run(accuracy, feed_dict = {x: batch_xs, y: batch_ys}))
        print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels}))
