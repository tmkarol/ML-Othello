#This will implement the AI for Othello

import numpy as numpy
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, Input
from main.py import GetPossibleMoves


def build_model():
    #multilayer model
    model = [ Conv2D(32,kernel_size=(3, 3), padding = 'same', activation='relu', input_shape = (28,28,1)),
    Conv2D(32,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(64,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(64,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(128,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(128,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(32,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(32,kernel_size=(1, 1), padding = 'same', activation='relu'),
    Dense((8,8), activation ='softmax')] #TODO sortof make this into an autoencoder to get proper output

    cnn_model = Sequential(cnn_layers)

    cnn_model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    cnn_model.fit(x_train.reshape(-1, 28, 28 ,1), y_train, epochs=1)

'''
Will yield in this order: 
1 black else 0
1 white else 0
1 free else 0
1 legal else 0
All 1 black win, all 0 tie, all -1 black lose
'''
def format_data():
    dataset = open("WTH_2004.txt", "rb").read()
    
    

print("Hello World")
    

