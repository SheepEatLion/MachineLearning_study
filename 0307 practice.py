#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

x_data = [1, 2, 3]
y_data = [3, 5, 7]

x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)


W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")


H = W * x + b


loss = tf.reduce_mean(tf.square(H-y))


train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(3000):
    tmp_train, loss_val = sess.run([train, loss], feed_dict={x:x_data, y:y_data})
    if step % 300 == 0:
        print("*** loss is {} ***".format(loss_val))

