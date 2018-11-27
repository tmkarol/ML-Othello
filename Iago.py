#This will implement the AI for Othello

# Just disables the warning, doesn't enable AVX/FMA because tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv3D, Reshape, Flatten
from tensorflow.keras import Sequential, Input


def build_model(X_train, y_test):
    #multilayer model

    model = [ Conv3D(32,kernel_size=(3, 3, 5), padding = 'same', activation='relu', input_shape = (8,8,5,1)),
    Conv3D(32,kernel_size=(3, 3, 5), padding = 'same', activation='relu'),
    Conv3D(64,kernel_size=(3, 3, 5), padding = 'same', activation='relu'),
    Conv3D(64,kernel_size=(3, 3, 5), padding = 'same', activation='relu'),
    Conv3D(128,kernel_size=(3, 3, 5), padding = 'same', activation='relu'),
    Conv3D(128,kernel_size=(3, 3, 5), padding = 'same', activation='relu'),
    Conv3D(32,kernel_size=(3, 3, 5), padding = 'same', activation='relu'),
    Conv3D(32,kernel_size=(1, 1, 5), padding = 'same', activation='relu'),
    Flatten(),
    Dense(192, activation ='softmax'),
    Reshape(target_shape=(8,8,3))]

    cnn_model = Sequential(model)
    cnn_model.summary()

    cnn_model.compile(optimizer="adam", loss='mse')
    cnn_model.fit(X_train.reshape(-1, 8, 8, 5, 1), y_train, epochs=1)


def format_data():
    dataset = open("WTH_2004.txt", "rb").read()

    return 0,0,0,0
    #return X_train, X_test, y_train, y_test

    

print("Hello World")
X_train, X_test, y_train, y_test = format_data()

X_train = np.zeros((13312,8,8,5))
y_train = np.ones((13312,8,8,3))

build_model(X_train, y_train)
    

