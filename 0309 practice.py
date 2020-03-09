#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("./data/ozone.csv", sep=",")
display(df.head())
display(df.shape)

df = df.dropna(how="any")
display(df.head())
display(df.shape)


df = df[["Ozone", "Solar.R", "Wind", "Temp"]]
display(df.head())


scaler = MinMaxScaler()
scaler.fit(df)

data = scaler.transform(df)

x_data = data[:,1:]
y_data = data[:, 0].reshape(-1, 1)
print(y_data.shape)

X = tf.placeholder(shape=[None, 3], dtype=tf.float32)
Y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

W = tf.Variable(tf.random_normal([3, 1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

H = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(H-Y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(3000):
    tmp_train, cost_val = sess.run([train, cost], feed_dict={X:x_data, Y:y_data})
    
    if step % 300 == 0:
        print("cost 값은 : {}".format(cost_val))

