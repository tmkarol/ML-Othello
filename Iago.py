#This will implement the AI for Othello

# Just disables the warning, doesn't enable AVX/FMA because tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Dense, Conv3D, Reshape, Flatten
from tensorflow.keras import Sequential

from main.py import GetPossibleMoves


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

'''
Will yield state arrays in this order: 
0: 1 black else 0
1: 1 white else 0
2: 1 free else 0
3: 1 legal else 0
4: All 1 black win, all 0 tie, all -1 black lose
'''
def format_data():
    dataset = open("WTH_2004.txt", "rb").read()
    # initialize X and y arrays
    # 13312 = number of games we have
    X = np.empty((13312, 5, 8, 8), int)
    y = np.empty((13312, 5, 8, 8), int)

    # For each game
    games = dataset.readlines()
    for game in games:
        moves = game.split()
        black_wins = moves[0] # Save how black did in the game
        board = np.zeroes((8, 8), int)
        board[3][3] = 'W'
        board[3][4] = 'B'
        board[4][3] = 'B'
        board[4][4] = 'W'
        # Walk through each game
        black_turn = true
        for move in moves[1:]:
            # Step through the game until final state is achieved
            if black_turn:
                player = "B"
            else:
                player = "W"
            x_move = int(move[1]) # column
            y_move = int(move[0]) # row
            # Check if move is legal. If it's not, that means the player skipped their turn.
            legal_moves = GetPossibleMoves(board, player)
            if (x_move, y_move) not in legal_moves:
                black_turn = not black_turn
                if black_turn:
                    player = "B"
                else:
                    player = "W"
            # Continue through the game
            flip = GetPiecesToFlip(board, x_move, y_move, player)
            board[y_move][x_move] = player
            board = FlipPieces(board, flip, player)
            # Put the states for black moves in X
            # Put the states for white moves in Y
            # TODO
            black_turn = not black_turn

    # Then! Split X and y into train and test

    return X_train, X_test, y_train, y_test

    
print("Hello World")
X_train, X_test, y_train, y_test = format_data()

X_train = np.zeros((13312,8,8,5))
y_train = np.ones((13312,8,8,3))

build_model(X_train, y_train)
    
