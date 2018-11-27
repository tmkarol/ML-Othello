#This will implement the AI for Othello

# Just disables the warning, doesn't enable AVX/FMA because tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Dense, Conv3D, Reshape, Flatten
from tensorflow.keras import Sequential

from main import GetPossibleMoves, GetPiecesToFlip, FlipPieces


def build_model(X_train, y_test):
    #multilayer model of convolutional 3D layers. Takes an (5,8,8) input. Outputs an (3,8,8)

    model = [ Conv3D(32,kernel_size=(3, 3, 5), padding = 'same', activation='relu', input_shape = (5,8,8,1)),
    Conv3D(32,kernel_size=(3, 3, 5), padding = 'same', activation='relu'),
    Conv3D(64,kernel_size=(3, 3, 5), padding = 'same', activation='relu'),
    Conv3D(64,kernel_size=(3, 3, 5), padding = 'same', activation='relu'),
    Conv3D(128,kernel_size=(3, 3, 5), padding = 'same', activation='relu'),
    Conv3D(128,kernel_size=(3, 3, 5), padding = 'same', activation='relu'),
    Conv3D(32,kernel_size=(3, 3, 5), padding = 'same', activation='relu'),
    Conv3D(32,kernel_size=(1, 1, 5), padding = 'same', activation='relu'),
    Flatten(),
    Dense(192, activation ='softmax'),
    Reshape(target_shape=(3,8,8))]

    cnn_model = Sequential(model)
    cnn_model.summary() #to figure out what is in the model

    cnn_model.compile(optimizer="adam", loss='mse')
    cnn_model.fit(X_train.reshape(-1, 5, 8, 8, 1), y_train, epochs=100)

    cnn_model.save("trained_model.h5")

'''
Will yield state arrays in this order: 
0: 1 black else 0
1: 1 white else 0
2: 1 free else 0
3: 1 legal else 0
4: All 1 black win, all 0 tie, all -1 black lose
'''
def format_data():
    dataset = open("WTH_2004.txt", "rb")
    output_X = open("WTH_dataset_X.txt","w")
    output_y = open("WTH_dataset_y.txt","w")
    # initialize X and y arrays
    # 13312 = number of games we have
    X = np.zeros((1, 5, 8, 8), int)
    y = np.zeros((1, 3, 8, 8), int)

    # For each game
    games = dataset.readlines()
    count = 0
    for game in games:
        
        if(count%100 == 0):
            print(count)
        count = count+1
        moves = game.split()

        black_win = moves[0] # Save how black did in the game    

        #build boardstate for game
        board = []
        for i in range(8):
            board.append([' '] * 8)

        board[3][3] = 'W'
        board[3][4] = 'B'
        board[4][3] = 'B'
        board[4][4] = 'W'

        #remove empty moves
        moves = [move for move in moves if move != b'-11']

        # Make a board state out of every move, assign to y if white to move, X if black to move
        black_turn = True
        for move in moves[1:]:
            # Step through the game until final state is achieved
            if black_turn:
                player = "B"
            else:
                player = "W"

            x_move = int(str(move)[2]) # column
            y_move = int(str(move)[3]) # row

            # Check if move is legal. If it's not, that means the player skipped their turn.
            legal_moves = GetPossibleMoves(board, player)
            if (x_move, y_move) not in legal_moves:
                black_turn = not black_turn
                if black_turn:
                    player = "B"
                else:
                    player = "W"

            #build data and targets
            if player == "B":
                tempX = np.zeros((1, 5, 8, 8), int) #same format as final data

                tempX[0][0][:][:] = (np.asarray(board) == "B").astype(int)
                tempX[0][1][:][:] = (np.asarray(board) == "W").astype(int)
                tempX[0][2][:][:] = np.logical_not(np.logical_xor(tempX[0][0][:][:],tempX[0][1][:][:]))
                for a in legal_moves:
                    tempX[0][3][a[0]][a[1]]
                tempX[0][4][:][:] = black_win

                output_X.write(tempX)
                output_X.write("\n")
                X = np.append(X,tempX,axis=0)

            elif player == "W":
                tempy = np.zeros((1,3,8,8), int)
                tempX[0][0][:][:] = (np.asarray(board) == "B").astype(int)
                tempX[0][1][:][:] = (np.asarray(board) == "W").astype(int)
                tempy[0][2][:][:] = np.logical_not(np.logical_xor(tempy[0][0][:][:],tempy[0][1][:][:]))

                output_y.write(tempy)
                output_y.write("\n")
                y = np.append(y,tempy,axis=0)


            # Continue through the game
            flip = GetPiecesToFlip(board, x_move, y_move, player)
            board[y_move][x_move] = player
            board = FlipPieces(board, flip, player)
            # Put the states for black moves in X
            # Put the states for white moves in Y
            # TODO
            black_turn = not black_turn

        # Put the states for that slice in X
        # Put the states for the next move in Y

    # Split X and y into train and test
    X_train = X[1:int(.8*X.shape[0])]
    X_test = X[int(.8*X.shape[0])+1:]
    y_train = y[1:int(.8*y.shape[0])]
    y_test = y[int(.8*y.shape[0])+1:]

    print(X[1])
    print(y[1])
    return X_train, X_test, y_train, y_test
    #return 0,0,0,0

    
print("Hello World")
X_train, X_test, y_train, y_test = format_data()

X_train = np.zeros((13312,5,8,8))
y_train = np.ones((13312,3,8,8))

#build_model(X_train, y_train)
    
