# -*- coding: utf-8 -*-
"""
@author: Ruiying

%% Imaging-based intelligent spectrometer on a plasmonic “rainbow” chip
% Dylan Tua1*, Ruiying Liu1*, Wenhong Yang2*, Lyu Zhou1, Haomin Song2, Leslie Ying1, Qiaoqiang Gan1,2 
%* These authors contribute equally to this work.
%1 Electrical Engineering, University at Buffalo, The State University of New York, Buffalo, NY 14260
%2 Material Science Engineering, Physical Science Engineering Division, King Abdullah University of Science and Technology, Thuwal 23955-6900, Saudi Arabia 

% Deep learning for ORD/polarization predict
% 12/18/2022
"""
import os
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Activation, Flatten, Conv2D, MaxPooling2D,Dropout, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io import loadmat as load

train_data = load('...path\\train_data.mat')
x_train = train_data['train_data']
train_X = x_train.reshape('num of train dataset','size of image','size of image',1)
#x_train = x_train.tolist()

train_label = load('...path\\train_label.mat')
y_train = train_label['train_label']
#y_train = y_train.tolist()

test_data = load('...path\\test_data.mat')
x_test = test_data['test_data']
test_X = x_test.reshape('num of test dataset','size of image','size of image',1)
#x_test = x_test.tolist()

test_label = load('...path\\test_label.mat')
y_test = test_label['test_label']

train_X = train_X.astype('float32')
test_X = test_X.astype('float32')

# total 9 different spectrum
train_Y_1 = y_train[:,:,0]
train_Y_2 = y_train[:,:,1]
train_Y_3 = y_train[:,:,2]
train_Y_4 = y_train[:,:,3]
train_Y_5 = y_train[:,:,4]
train_Y_6 = y_train[:,:,5]
train_Y_7 = y_train[:,:,6]
train_Y_8 = y_train[:,:,7]
train_Y_9 = y_train[:,:,8]
test_Y_one_hot_1 = y_test[:,:,0]
test_Y_one_hot_2 = y_test[:,:,1]
test_Y_one_hot_3 = y_test[:,:,2]
test_Y_one_hot_4 = y_test[:,:,3]
test_Y_one_hot_5 = y_test[:,:,4]
test_Y_one_hot_6 = y_test[:,:,5]
test_Y_one_hot_7 = y_test[:,:,6]
test_Y_one_hot_8 = y_test[:,:,7]
test_Y_one_hot_9 = y_test[:,:,8]


def singleNet(input_layer):
    X = Conv2D(filters=64, kernel_size = (3,3), padding = 'same', activation = 'elu')(input_layer)
    X = MaxPooling2D(pool_size = (2,2), strides = 2)(X)
    X = Conv2D(filters=128, kernel_size = (3,3), padding = 'same', activation = 'elu')(X)
    X = MaxPooling2D(pool_size = (2,2), strides = 2)(X)
    X = Conv2D(filters=256, kernel_size = (3,3), padding = 'same', activation = 'elu')(X)
    X = MaxPooling2D(pool_size = (2,2), strides = 2)(X)
    X = Flatten()(X)
    X = Dense(512, activation = 'elu')(X)
    X = Dropout(0.6)(X)
    X = Dense(64, activation = 'elu')(X)
    X = Dropout(0.2)(X)
    output = Dense(30, activation = 'softmax')(X)
    return output
    
    
def doubleNet():
    input_layer = Input(shape = (53, 53, 1))
    X1 = singleNet(input_layer)
    X2 = singleNet(input_layer)
    X3 = singleNet(input_layer)
    X4 = singleNet(input_layer)
    X5 = singleNet(input_layer)
    X6 = singleNet(input_layer)
    X7 = singleNet(input_layer)
    X8 = singleNet(input_layer)
    X9 = singleNet(input_layer)
    model = Model(input_layer, [X1,X2,X3,X4,X5,X6,X7,X8,X9], name = 'doubleNet')
    return model

model = doubleNet()


model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.fit(train_X, [train_Y_1,train_Y_2,train_Y_3,train_Y_4,train_Y_5,train_Y_6,train_Y_7,train_Y_8,train_Y_9], batch_size=64, epochs=10,shuffle=True)

model.save('my_model3.h5')


predictions = model.predict(test_X)
sio.savemat("prediction.mat", {"prediction": predictions})
