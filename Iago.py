#This will implement the AI for Othello

# Just disables the warning, doesn't enable AVX/FMA because tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.layers import Dense, Conv2D, Reshape, Flatten
from tensorflow.keras import Sequential

from main import GetPossibleMoves, GetPiecesToFlip, FlipPieces


def build_model(X_train, y_train):
    #multilayer model of convolutional 3D layers. Takes an (4,8,8) input. Outputs an (3,8,8)

    model = [ Conv2D(32,kernel_size=(3, 3), padding = 'same', activation='relu', input_shape = (4,8,8)),
    Conv2D(32,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(64,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(64,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(128,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(128,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(256,kernel_size=(3, 3), padding = 'same', activation='relu'),
    Conv2D(256,kernel_size=(1, 1), padding = 'same', activation='relu'),
    Flatten(),
    Dense(192, activation ='softmax'),
    Reshape(target_shape=(3,8,8))]

    cnn_model = Sequential(model)
    cnn_model.summary() #to figure out what is in the model

    cnn_model.compile(optimizer="adam", loss='mse', metrics=['accuracy'])
    cnn_model.fit(X_train.reshape(-1, 4, 8, 8), y_train, epochs=5)

    cnn_model.save("trained_model.h5")

    return cnn_model


#TODO make model predict the next move using argmax over only legal move squares
def evaluate_model(model,X,y):
    score = model.evaluate(X,y)
    print(f"\nThe simple model achieves an accuracy of {score[1]*100:.2f}% on the test data.")

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
    X = []
    y = []

    # For each game
    games = dataset.readlines()
    count = 0

    for game in games:
        
        if len(X) > len(y):
            X.pop()
        if count%100 == 0:
            print(count)
        count = count + 1
        moves = game.split() # Separate by spaces

        if moves[0] == -1:
            continue #throw out losses    

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
                    tempX = np.zeros((4,8,8), int) #same format as final data

                    tempX[0][:][:] = (np.asarray(board) == "B").astype(int)
                    tempX[1][:][:] = (np.asarray(board) == "W").astype(int)
                    tempX[2][:][:] = np.logical_not(np.logical_xor(tempX[0][:][:],tempX[1][:][:]))
                    for a in legal_moves:
                        tempX[3][a[0]][a[1]]

                    tempy = np.zeros((3,8,8), int)
                    tempy[0][:][:] = (np.asarray(board) == "B").astype(int)
                    tempy[1][:][:] = (np.asarray(board) == "W").astype(int)
                    tempy[2][:][:] = np.logical_not(np.logical_xor(tempy[0][:][:],tempy[1][:][:]))

                    y.append(tempy)
                    X.append(tempX)

                else:
                    player = "W"
            else:
                #Else, it's a legal move
                #build data and targets
                if player == "B":
                    tempX = np.zeros((4,8,8), int) #same format as final data

                    tempX[0][:][:] = (np.asarray(board) == "B").astype(int)
                    tempX[1][:][:] = (np.asarray(board) == "W").astype(int)
                    tempX[2][:][:] = np.logical_not(np.logical_xor(tempX[0][:][:],tempX[1][:][:]))
                    for a in legal_moves:
                        tempX[3][a[0]][a[1]]

                    X.append(tempX)

                elif player == "W":
                    tempy = np.zeros((3,8,8), int)
                    tempy[0][:][:] = (np.asarray(board) == "B").astype(int)
                    tempy[1][:][:] = (np.asarray(board) == "W").astype(int)
                    tempy[2][:][:] = np.logical_not(np.logical_xor(tempy[0][:][:],tempy[1][:][:]))

                    y.append(tempy)


            # Continue through the game
            flip = GetPiecesToFlip(board, x_move, y_move, player)
            board[y_move][x_move] = player
            board = FlipPieces(board, flip, player)
            black_turn = not black_turn

    X = np.asarray(X)
    y = np.asarray(y)
    
    print(X.shape)
    print(y.shape)

    #save everything
    #open with np.loadtxt('WTH_dataset_X.txt').reshape((271971, 5, 8, 8))
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    output_X.write('# Data shape: {0}\n'.format(X.shape))
    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for sample in X:
        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.
        for sample_slice in sample:
            np.savetxt(output_X, sample_slice, fmt='%d')

        # Writing out a break to indicate different slices...
            output_X.write('# New slice\n')
        output_X.write('# New sample\n')

    output_y.write('# Data shape: {0}\n'.format(y.shape))
    # Iterating through a ndimensional array produces slices along
    # the last axis. This is equivalent to data[i,:,:] in this case
    for sample in y:
        # The formatting string indicates that I'm writing out
        # the values in left-justified columns 7 characters in width
        # with 2 decimal places.
        for sample_slice in sample:
            np.savetxt(output_y, sample_slice, fmt='%d')

        # Writing out a break to indicate different slices...
            output_y.write('# New slice\n')
        output_y.write('# New sample\n')

    # Split X and y into train and test
    X_train = X[:int(.8*X.shape[0])]
    X_test = X[int(.8*X.shape[0])+1:]
    y_train = y[:int(.8*y.shape[0])]
    y_test = y[int(.8*y.shape[0])+1:]

    return X_train, X_test, y_train, y_test
    #return 0,0,0,0


#TODO check if dataset files exist and load if they do. Remake if they dont

X_train, X_test, y_train, y_test = format_data()

print(X_train.shape)
print(y_train.shape)

model = build_model(X_train, y_train)
evaluate_model(model, X_test, y_test)
