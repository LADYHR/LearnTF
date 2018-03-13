# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:35:21 2018

@author: Hua Rui
"""
#导入MNIST数据集
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
import tensorflow as tf
#构建变量、参数
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#定义softmax模型
y = tf.nn.softmax(tf.matmul(x,W) + b)
#计算损失函数
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#选择训练优化算法
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
#训练模型
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})

#评估模型
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))#tf.cast做数据类型转换
print (sess.run(accuracy,feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
sess.close()#关闭会话
