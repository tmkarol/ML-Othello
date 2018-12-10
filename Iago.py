#This will implement the AI training for Othello

# This disables the warnings for parts of tensorflow, doesn't enable AVX/FMA
import os
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Imports for the model
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dense, Conv2D, Reshape, Flatten, MaxPooling2D
from tensorflow.keras import Sequential

# Imported for gridsearch stage but no longer required
#from sklearn.model_selection import GridSearchCV

# Imports from the game
#from main import GetPossibleMoves, GetPiecesToFlip, FlipPieces

# This creates the model using Keras. The defaults were set using a gridsearch over each parameter
# It can be wrapped with a scikit learn wrapper to use with gridsearch
def create_model(optimizer="Adadelta",activation = "relu", neurons_a = 32, neurons_b = 64, neurons_c = 128, padding = "same", loss = "categorical_crossentropy", kernel_sz = (3,3)):
    #multilayer model of convolutional 3D layers. Takes an (8,8,4) input. Outputs a flattened (8,8,1) board.

    model = [ Conv2D(neurons_c,kernel_size=(5,5), padding = padding, activation=activation, input_shape = (8,8,4)),
    Conv2D(neurons_c,kernel_size=kernel_sz, padding = padding, activation=activation),
    Conv2D(neurons_b,kernel_size=kernel_sz, padding = padding, activation=activation),
    Conv2D(neurons_b,kernel_size=kernel_sz, padding = padding, activation=activation),
    Conv2D(neurons_a,kernel_size=kernel_sz, padding = padding, activation=activation),
    Conv2D(neurons_a,kernel_size=kernel_sz, padding = padding, activation=activation),
    Conv2D(neurons_b,kernel_size=kernel_sz, padding = padding, activation=activation),
    Conv2D(neurons_b,kernel_size=kernel_sz, padding = padding, activation=activation),
    Conv2D(neurons_c,kernel_size=kernel_sz, padding = padding, activation=activation),
    Conv2D(neurons_c,kernel_size=(5,5), padding = padding, activation=activation),
    Flatten(),
    Dense(64, activation ='softmax')]

    cnn_model = Sequential(model)

    cnn_model.summary() # For debugging purposes

    cnn_model.compile(optimizer=optimizer, loss=loss, metrics=['mse','accuracy'])

    return cnn_model

# This tests the models performance after training
def evaluate_model(model,X_test,y_test):
    score = model.evaluate(X_test,y_test)
    print(score)
    print(f"\nThe simple model achieves an mse of {score[1]:.2f} on the test data.")
    print(f"\nThe simple model achieves an accuracy of {score[2]*100:.2f} on the test data.")
    return score[2]*100

# This function takes a .txt file full of move sequences and converts it into (8,8,4) board state arrays.
# The function only makes state arrays for games black wins.
# Takes a long time to run due to dataset size. We have included it's output files in out submission.
# Will yield state arrays in the format: 
# Layer 0: 1 black else 0
# Layer 1: 1 white else 0
# Layer 2: 1 free else 0
# Layer 3: 1 legal else 0
def format_data(filename, outfileX, outfiley):

    from main import GetPossibleMoves, GetPiecesToFlip, FlipPieces
    
    # Open our reading and writing files
    dataset = open(filename, "rb")
    output_X = open(outfileX,"w")
    output_y = open(outfiley,"w")
    # initialize X and y arrays
    X = []
    y = []

    # For each game
    games = dataset.readlines()
    count = 0

    for game in games:
        #Pop off each final move if white was the winner of the game
        if len(X) > len(y):
            X.pop()
        if count%100 == 0:
            print(count)
        count = count + 1
        moves = game.split() # Separate each move

        # First number in each game shows the winner. We throw out losses for black
        if moves[0] == -1:
            continue 

        # build initial boardstate for game as it is identical for each game
        board = []
        for i in range(8):
            board.append([' '] * 8)

        board[3][3] = 'W'
        board[3][4] = 'B'
        board[4][3] = 'B'
        board[4][4] = 'W'

        # For games that end in fewer than 60 moves, the maximum number of moves,
        # we throw out the leftover turns as each game is stored in a constant size.
        moves = [move for move in moves if move != b'-11']

        # Make a board state out of every move, assign to y if white to move, X if black to move
        black_turn = True
        for move in moves[1:]:
            if black_turn:
                player = "B"
            else:
                player = "W"

            x_move = int(str(move)[2]) # column
            y_move = int(str(move)[3]) # row

            # Check if move is legal. If it's not, that means the player skipped their turn
            # because they had no legal mvoes
            legal_moves = GetPossibleMoves(board, player)
            if (x_move, y_move) not in legal_moves:
                black_turn = not black_turn
                #If white was skipped we make it blacks turn and build an X,y pair.
                if black_turn:
                    player = "B"
                    tempX = np.zeros((8,8,4), int)

                    tempX[:,:,0] = (np.asarray(board) == "B").astype(int)   # Black positions
                    tempX[:,:,1] = (np.asarray(board) == "W").astype(int)   # White positions
                    tempX[:,:,2] = np.logical_not(np.logical_xor(tempX[:,:,0],tempX[:,:,1])) # Empty spaces
                    for a in legal_moves:
                        tempX[a[1],a[0],3] = 1  # Legal moves

                    tempy = np.zeros((8,8,1), int)
                    tempy[:,:,0] = np.logical_and((np.asarray(board) == "B").astype(int), X[len(X)-2][:,:,3]) #Where black moved

                    y.append(tempy)
                    X.append(tempX)

                else:
                    # If black skipped thier turn we do nothing because the AI couldn't make a decision anyway
                    player = "W"
            else:
                #If it's a legal move
                #build X if it is black to move
                if player == "B":
                    tempX = np.zeros((8,8,4), int)

                    tempX[:,:,0] = (np.asarray(board) == "B").astype(int)   # Black positions
                    tempX[:,:,1] = (np.asarray(board) == "W").astype(int)   # White positions
                    tempX[:,:,2] = np.logical_not(np.logical_xor(tempX[:,:,0],tempX[:,:,1])) # Empty spaces
                    for a in legal_moves:
                        tempX[a[1],a[0],3] = 1  # Legal moves

                    X.append(tempX)

                elif player == "W":
                    tempy = np.zeros((8,8,1), int)
                    tempy[:,:,0] = np.logical_and((np.asarray(board) == "B").astype(int), X[len(X)-2][:,:,3]) # Where black moved

                    y.append(tempy)


            # Continue to the next turn of the game
            flip = GetPiecesToFlip(board, x_move, y_move, player)
            board[y_move][x_move] = player
            board = FlipPieces(board, flip, player)
            black_turn = not black_turn

    X = np.asarray(X)
    y = np.asarray(y)
    
    # save everything
    # open with np.loadtxt('WTH_dataset_X.txt').reshape((#samples, 8, 8, 4))
    # I'm writing a header here just for the sake of readability
    # Any line starting with "#" will be ignored by numpy.loadtxt
    output_X.write('#'+str(len(X)))
    output_X.write('\n# Data shape: {0}\n'.format(X.shape))
    for sample in X:
        for sample_slice in sample:
            np.savetxt(output_X, sample_slice, fmt='%d')

        # Writing out a break to indicate different slices...
            output_X.write('# New slice\n')
        output_X.write('# New sample\n')

    # Save y with the same format
    # open with np.loadtxt('WTH_dataset_{year}_y.txt').reshape((len(y),64))
    output_y.write('#'+str(len(y)))
    output_y.write('\n# Data shape: {0}\n'.format(y.shape))
    
    for sample in y:
        for sample_slice in sample:
            np.savetxt(output_y, sample_slice, fmt='%d')
            output_y.write('# New slice\n')
        output_y.write('# New sample\n')

    y = y.reshape(len(y), 64)

    return X, y

# Helper for reading .wtb files from WTHOR database
# Output is the game result for black followed by the move list
def bytes_to_int(bytes):
    return int.from_bytes(bytes, byteorder='little')

# Converts .wtb files to .txt movelists
def load_wtb_file(filename):
    # open our reading and writing files
    dataset = open(filename, "rb").read()
    output = open(filename[:len(filename)-3]+"txt","w")

    # The .wtb header is 16 bytes with bytes 3-6 storing the number of games in the file 
    header = dataset[:15]
    numberOfRecords = bytes_to_int(header[4:7])

    # The games are listed after the header until the end of the file
    body = dataset[16:]

    # print(numberOfRecords) # sanity check

    for x in range(numberOfRecords):
        '''
        Each record is 68 bytes and contains:

        Label                       Size(Bytes)     Type

        tournament label number     2               Word
        Player number Black         2               Word
        Player number White         2               Word
        Real score                  1               Byte
        Theoretical score           1               Byte
        List of moves               60              Byte[]
        '''
        #Take each individual game
        start_index = x * 68
        end_index = start_index + 68
        record = body[start_index:end_index]

        # Determine who wone the game and write it ot file
        if record[6] > 32:      # Black won
            output.write(str(1))
        elif record[6] == 32:   # Tie
            output.write(str(0))
        else:                   # Black Lost
            output.write(str(-1))
        output.write(' ')

        # Parse the move list and write ot file
        for i in range(8,68):
            move = str(record[i]-11).zfill(2)
            output.write(move)
            output.write(' ')
        output.write('\n')

def load_dataset(year):
    rawfilename = f"WTH_{year}.wtb"
    parsedfilename = f"WTH_{year}.txt"
    datasetfilenameX = f"WTH_dataset_{year}_X.txt"
    datasetfilenamey = f"WTH_dataset_{year}_y.txt"
    # If the dataset does not exist, make them from the raw .txt files
    # If it does just load from file
    if (not os.path.isfile(datasetfilenameX)) or (not os.path.isfile(datasetfilenamey)) or os.stat(datasetfilenameX).st_size == 0 or os.stat(datasetfilenamey).st_size == 0:
        # If we don't have the parsed movelist, make them from the .wtb files then build.
        # If we have the parsed move list just build
        if (not os.path.isfile(parsedfilename)) or os.stat(parsedfilename).st_size == 0:
            print("Building .txt from .wtb")
            load_wtb_file(rawfilename)
        
        print("Building dataset from .txt")
        X, y= format_data(parsedfilename,datasetfilenameX,datasetfilenamey)
    else:
        print("Load data from file")
        X = open(datasetfilenameX)
        szx = int(X.readline()[1:])
        X.close()
        y = open(datasetfilenamey)
        szy = int(y.readline()[1:])
        y.close()
        print(f"Loading {szx} features and targets, please wait.")

        X = np.loadtxt(datasetfilenameX).reshape((szx, 8, 8, 4))
        y = np.loadtxt(datasetfilenamey).reshape((szy, 64))

        # Make train and test sets with 20% going to testing
    X_train = X[:int(.8*X.shape[0])]        
    X_test = X[int(.8*X.shape[0])+1:]
    y_train = y[:int(.8*y.shape[0])]
    y_test = y[int(.8*y.shape[0])+1:]

    # We used a reduced dataset while running our grid search
    #X_train = X[:5000]
    #y_train = y[:5000].reshape(5000,64)

    return X_train,y_train,X_test,y_test
'''
Function to train a model. This function will create or load new files as necessary.
'''
def train_model():
    # Set the dataset year here:
    year = int(input("Which dataset year? "))

    X_train,y_train,X_test,y_test = load_dataset(year)

    #print(X_train.shape) # Sanity checks
    #print(y_train.shape)

    # We use the KerasClassifier wrapper to implement gridsearch but it is extraneous 
    # for training outside of the grid search because the wrapper cannot use Keras'
    # built in save function.
    #model = KerasClassifier(build_fn=create_model, verbose=2)

    # define the grid search parameters
    #batch_size = [1,10]
    #epochs = [1,2,5]
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

    # build model
    model = create_model()
    print("Begin Fit")
    grid_result = model.fit(X_train, y_train, epochs=10, batch_size=100)


    #print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #means = grid_result.cv_results_['mean_test_score']
    #stds = grid_result.cv_results_['std_test_score']
    #params = grid_result.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
    #    print("%f (%f) with: %r" % (mean, stdev, param))

    # Finally we test and save our model with it's test score
    score = evaluate_model(model, X_test, y_test)
    model.save(f"model{score:.2f}.h5")