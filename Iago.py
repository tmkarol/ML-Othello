#This will implement the AI for Othello

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras import Model, Input


def build_model(X_train, y_test):
    #multilayer model
    inputs = tf.keras.Input(shape=X_train[0].shape)

    model = [ Conv2D(32,kernel_size=(3, 3), padding = 'same', activation='relu', input_shape = (8,8,5)),
    Conv2D(32,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(64,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(64,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(128,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(128,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(32,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(32,kernel_size=(1, 1), padding = 'same', activation='relu'),
    Dense(, activation ='softmax')] #TODO sortof make this into an autoencoder to get proper output

    cnn_model = Sequential(model)

    cnn_model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #cnn_model.fit(x_train.reshape(-1, 28, 28 ,1), y_train, epochs=1)


def format_data():
    dataset = open("WTH_2004.txt", "rb").read()


    return X_train, X_test, y_train, y_test

    

print("Hello World")
X_train, X_test, y_train, y_test = format_data()

build_model(X_train, y_train)
    

