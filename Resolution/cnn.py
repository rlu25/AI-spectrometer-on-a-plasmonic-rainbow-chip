# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 00:17:58 2019

@author: Ruiying

%% Imaging-based intelligent spectrometer on a plasmonic “rainbow” chip
% Dylan Tua1*, Ruiying Liu1*, Wenhong Yang2*, Lyu Zhou1, Haomin Song2, Leslie Ying1, Qiaoqiang Gan1,2 
%* These authors contribute equally to this work.
%1 Electrical Engineering, University at Buffalo, The State University of New York, Buffalo, NY 14260
%2 Material Science Engineering, Physical Science Engineering Division, King Abdullah University of Science and Technology, Thuwal 23955-6900, Saudi Arabia 

% Deep learning for resolution
% 12/18/2022
"""




import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow.compat.v2 as tf
# import tensorflow as tf
import scipy.io as sio
from scipy.io import loadmat as load

 
step = 5000
learning_rate = 0.001
batch = 128
logs_train_dir = 'data'  #logs save path



train_data = load('...path\\train_data.mat')
x_train = train_data['train_data']
x_train = x_train.reshape('num of train','size of pattern','size of pattern',1)


train_label = load('...path\\train_label.mat')
y_train = train_label['train_label']


test_data = load('...path\\test_data.mat')
x_test = test_data['test_data']
x_test = x_test.reshape('num of test','size of pattern','size of pattern',1)


test_label = load('...path\\test_label.mat')
y_test = test_label['test_label']


# 数据数据集
x = tf.placeholder(dtype=tf.float32, shape=[None, 'size of pattern','size of pattern',1], name='x-input');

# 训练数据集中的label
y = tf.placeholder(dtype=tf.float32, shape=[None, 'size of last layer'], name='y-input');

w1 = tf.Variable(tf.random_normal([2048,1024]))
w2 = tf.Variable(tf.random_normal([1024,'size of last layer']))

 
b1 = tf.Variable(tf.random_normal([1024]))
b2 = tf.Variable(tf.random_normal(['size of last layer'])) 

 


def CNN(X):
    
    conv0 = tf.layers.conv2d(X, 64, 3, activation=tf.nn.relu,padding='valid')
    pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2],padding='valid')
    conv1 = tf.layers.conv2d(pool0, 128, 3, activation=tf.nn.relu,padding='valid')
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2],padding='valid')
    conv2 = tf.layers.conv2d(pool1, 256, 3, activation=tf.nn.relu,padding='valid')
    pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2],padding='valid')
    conv3 = tf.layers.conv2d(pool2, 512, 3, activation=tf.nn.relu,padding='valid')
    pool3 = tf.layers.max_pooling2d(conv3, [2, 2], [2, 2],padding='valid')
    flatten = tf.layers.flatten(pool3)
    
    
    f1 = tf.layers.dense(flatten, 2048, activation=tf.nn.relu)
    fc2=tf.nn.relu(tf.add(tf.matmul(fc1,w1),b1))
    
    fc3=tf.nn.relu(tf.add(tf.matmul(fc2,w2),b2))
    dropout_fc = tf.layers.dropout(fc3, 0.25)

    net=dropout_fc
 
    return net
 
    
net_out = CNN(x)
 
pre = tf.nn.softmax(net_out)
 
 
 

 
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-net_out),reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
 
train_op = optimizer.minimize(loss)

 
correct_pre = tf.equal(tf.argmax(pre,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pre,tf.float32))
 
init = tf.global_variables_initializer()
saver = tf.train.Saver()


   
with tf.Session() as sess:
    
    sess.run(init)

    for i in range(1,step+1):
        
       
        sess.run(train_op, feed_dict={x: x_train, y: y_train})
        if i % 100 == 0 or i == 1:
            l, acc = sess.run([loss, accuracy], feed_dict={x: x_train,
                                                                 y: y_train})
            print("Step " + str(i) + ", Minibatch Loss= " + "{:.4f}".format(l) + ", Training Accuracy= " + "{:.3f}".format(acc))
            
    checkpoint_path = os.path.join(logs_train_dir, 'thing.ckpt')
    saver.save(sess, checkpoint_path)        
    print("Optimization Finished!")
 

    prediction_value = sess.run(net_out, feed_dict={x: x_test})
    
    sio.savemat("prediction.mat", {"prediction": prediction_value})
    
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: x_test,
                                      y: y_test}))