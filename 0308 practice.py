#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

x_data = [ [73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70] ]
y_data = [ [152], [185], [180], [196], [142]]

X = tf.placeholder(shape=[None, 3], dtype=tf.float32)
Y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

W = tf.Variable(tf.random_normal([3, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

H = tf.matmul(X, W) + b

loss = tf.reduce_mean(tf.square(H-Y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(3000):
    temp_train, loss_val = sess.run([train, loss], feed_dict={X:x_data, Y:y_data})
    if step % 300 == 0:
        print("*** loss is : {}".format(loss_val))

