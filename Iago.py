#This will implement the AI for Othello

# Just disables the warning, doesn't enable AVX/FMA because tensorflow
import os
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from tensorflow.keras.layers import Dense, Conv2D, Reshape, Flatten
from tensorflow.keras import Sequential

from main import GetPossibleMoves, GetPiecesToFlip, FlipPieces


def create_model(optimizer="Adadelta",activation = "relu", neurons_a = 32, neurons_b = 64, neurons_c = 128, padding = "same", loss = "categorical_crossentropy", kernel_sz = (3,3)):
    #multilayer model of convolutional 3D layers. Takes an (8,8,4) input. Outputs an (8,8,1)

    model = [ Conv2D(neurons_a,kernel_size=kernel_sz, padding = padding, activation=activation, input_shape = (8,8,4)),
    Conv2D(neurons_a,kernel_size=kernel_sz, padding = padding, activation=activation),
    Conv2D(neurons_b,kernel_size=kernel_sz, padding = padding, activation=activation),
    Conv2D(neurons_b,kernel_size=kernel_sz, padding = padding, activation=activation),
    Conv2D(neurons_c,kernel_size=kernel_sz, padding = padding, activation=activation),
    Conv2D(neurons_c,kernel_size=kernel_sz, padding = padding, activation=activation),
    Conv2D(neurons_b,kernel_size=kernel_sz, padding = padding, activation=activation),
    Conv2D(neurons_a,kernel_size=(1, 1), padding = padding, activation=activation),
    Flatten(),
    Dense(64, activation ='softmax')]
    #Reshape(target_shape=(1,8,8))]

    cnn_model = Sequential(model)
    cnn_model.summary() #to figure out what is in the model

    cnn_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return cnn_model


def evaluate_model(model,X_test,y_test):
    score = model.evaluate(X_test,y_test)
    print(f"\nThe simple model achieves an accuracy of {score[1]*100:.2f}% on the test data.")



'''
Will yield state arrays in this order: 
0: 1 black else 0
1: 1 white else 0
2: 1 free else 0
3: 1 legal else 0
Gets rid of games black loses
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
                    tempX = np.zeros((8,8,4), int) #same format as final data

                    tempX[:,:,0] = (np.asarray(board) == "B").astype(int)
                    tempX[:,:,1] = (np.asarray(board) == "W").astype(int)
                    tempX[:,:,2] = np.logical_not(np.logical_xor(tempX[:,:,0],tempX[:,:,1]))
                    for a in legal_moves:
                        tempX[a[1],a[0],3] = 1

                    tempy = np.zeros((8,8,1), int)
                    tempy[:,:,0] = np.logical_and((np.asarray(board) == "B").astype(int), X[len(X)-2][:,:,3])

                    y.append(tempy)
                    X.append(tempX)

                else:
                    player = "W"
            else:
                #Else, it's a legal move
                #build data and targets
                if player == "B":
                    tempX = np.zeros((8,8,4), int) #same format as final data

                    tempX[:,:,0] = (np.asarray(board) == "B").astype(int)
                    tempX[:,:,1] = (np.asarray(board) == "W").astype(int)
                    tempX[:,:,2] = np.logical_not(np.logical_xor(tempX[:,:,0],tempX[:,:,1]))
                    for a in legal_moves:
                        tempX[a[1],a[0],3] = 1

                    X.append(tempX)

                elif player == "W":
                    tempy = np.zeros((8,8,1), int)
                    tempy[:,:,0] = np.logical_and((np.asarray(board) == "B").astype(int), X[len(X)-2][:,:,3])

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
    #open with np.loadtxt('WTH_dataset_X.txt').reshape((271971, 8, 8, 4))
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    output_X.write('# Data shape: {0}\n'.format(X.shape))
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
    
    for sample in y:
        for sample_slice in sample:
            np.savetxt(output_y, sample_slice, fmt='%d')
            output_y.write('# New slice\n')
        output_y.write('# New sample\n')

    # Split X and y into train and test
    X_train = X[:int(.8*X.shape[0])]
    X_test = X[int(.8*X.shape[0])+1:]
    y_train = y[:int(.8*y.shape[0])]
    y_test = y[int(.8*y.shape[0])+1:]

    return X_train, X_test, y_train, y_test

if (not os.path.isfile('WTH_dataset_X.txt')) or (not os.path.isfile('WTH_dataset_y.txt')) or os.stat('WTH_dataset_X.txt').st_size == 0 or os.stat('WTH_dataset_y.txt').st_size == 0:
    print("Building data")
    X_train, X_test, y_train, y_test = format_data()
else:
    print("Load data from file")
    X = np.loadtxt('WTH_dataset_X.txt').reshape((271971, 8, 8, 4))
    y = np.loadtxt('WTH_dataset_y.txt').reshape((271971, 8, 8, 1)).reshape(271971,64)
    X_train = X[:int(.8*X.shape[0])]        
    X_test = X[int(.8*X.shape[0])+1:]
    y_train = y[:int(.8*y.shape[0])]
    y_test = y[int(.8*y.shape[0])+1:]

#X_train = X[:5000]
#y_train = y[:5000].reshape(5000,64)

print(X_train.shape)
print(y_train.shape)

#model = KerasClassifier(build_fn=create_model, verbose=2)
model = create_model()

# define the grid search parameters
#batch_size = [1]
#epochs = [2]
#optimizer = ["SGD", "Adadelta", "Adam"]
#activation = ['relu', 'sigmoid']
#neurons_a = [16,32]
#neurons_b = [64,128]
#neurons_c = [128,256]
#padding = ["same"]
#loss = ["categorical_crossentropy", 'mean_squared_error','categorical_hinge']
#kernel_sz = [(3,3)]

#param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, activation=activation,neurons_a=neurons_a,neurons_b=neurons_b,neurons_c=neurons_c,padding=padding,loss=loss)
#grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=2,verbose = 1)

print("Begin Fit")
grid_result = model.fit(X_train, y_train, epochs=7, batch_size=100)

"""
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
"""

model.save("model.h5")
#joblib.dump(model, 'model.pkl') 
evaluate_model(model, X_test, y_test)