# -*- coding: utf-8 -*-
"""

@author: Ruiying
%% Imaging-based intelligent spectrometer on a plasmonic “rainbow” chip
% Dylan Tua1*, Ruiying Liu1*, Wenhong Yang2*, Lyu Zhou1, Haomin Song2, Leslie Ying1, Qiaoqiang Gan1,2 
%* These authors contribute equally to this work.
%1 Electrical Engineering, University at Buffalo, The State University of New York, Buffalo, NY 14260
%2 Material Science Engineering, Physical Science Engineering Division, King Abdullah University of Science and Technology, Thuwal 23955-6900, Saudi Arabia 

% Deep learning for 1D pattern
% 12/18/2022
"""




import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.io as sio
from scipy.io import loadmat as load

 
step =5000
learning_rate = 0.001
batch = 128
logs_train_dir = 'data'  #logs save path



train_data = load('...path\\train_data.mat')
x_train = train_data['train_data']
x_train = x_train.reshape(500,187,34,1)


train_label = load('...path\\train_label.mat')
y_train = train_label['train_label']


test_data = load('...path\\test_data.mat')
x_test = test_data['test_data']
x_test = x_test.reshape(488,187,34,1)


test_label = load('...path\\test_label.mat')
y_test = test_label['test_label']



# data
x = tf.placeholder(dtype=tf.float32, shape=[None, 187,34,1], name='x-input');

# label
y = tf.placeholder(dtype=tf.float32, shape=[None, 600], name='y-input');

 
w1 = tf.Variable(tf.random_normal([2048,1024]))
w2 = tf.Variable(tf.random_normal([1024,600]))

 
b1 = tf.Variable(tf.random_normal([1024]))
b2 = tf.Variable(tf.random_normal([600]))

 
def CNN_network(X):

    conv0 = tf.layers.conv2d(X, 20, 5, activation=tf.nn.relu,padding='same')
    pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2],padding='same')
    conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu,padding='same')
    pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2],padding='same')
    conv2 = tf.layers.conv2d(pool1, 64, 3, activation=tf.nn.relu,padding='same')
    pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2],padding='same')
    conv3 = tf.layers.conv2d(pool2, 128, 3, activation=tf.nn.relu,padding='same')
    pool3 = tf.layers.max_pooling2d(conv3, [2, 2], [2, 2],padding='valid')

    flatten = tf.layers.flatten(pool3)
    
    
    fc1 = tf.layers.dense(flatten, 2048, activation=tf.nn.relu)
    fc2=tf.nn.relu(tf.add(tf.matmul(fc1,w1),b1))
    
    fc3=tf.nn.relu(tf.add(tf.matmul(fc2,w2),b2))
    
    dropout_fc = tf.layers.dropout(fc3, 0.25)

    net=dropout_fc

    return net
 
    
net_out = CNN_network(x)
 
pre = tf.nn.softmax(net_out)
 
 
 

 
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y-net_out),reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
 
train_op = optimizer.minimize(loss)

 
correct_pre = tf.equal(tf.argmax(pre,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pre,tf.float32))
 
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()
# save loss and acc during training
loss_plot=np.zeros(int(step/10)+1) 
acc_plot=np.zeros(int(step/10)+1)   

with tf.Session() as sess:
    
    sess.run(init)

    summary_writer = tf.summary.FileWriter(logs_train_dir, graph=tf.get_default_graph())
    for i in range(1,step+1):
        
       
        sess.run(train_op, feed_dict={x: x_train, y: y_train})
        if i % 10 == 0 or i == 1:
            l, acc = sess.run([loss, accuracy], feed_dict={x: x_train,
                                                                 y: y_train})
            loss_plot[int(i/10)]=l
            acc_plot[int(i/10)]=acc
            print("Step " + str(i) + ", Minibatch Loss= " + "{:.4f}".format(l) + ", Training Accuracy= " + "{:.3f}".format(acc))
            
    checkpoint_path = os.path.join(logs_train_dir, 'thing.ckpt')
    saver.save(sess, checkpoint_path)        
    print("Optimization Finished!")
 

  
    prediction_value = sess.run(net_out, feed_dict={x: x_test})
    
    sio.savemat("prediction.mat", {"prediction": prediction_value})
    sio.savemat("loss.mat", {"loss": loss_plot})
    sio.savemat("acc.mat", {"acc": acc_plot})
    
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: x_test,
                                      y: y_test}))
