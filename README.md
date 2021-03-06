IAGO: THE OTHELLO AI
====================

FILES
-----

main.py - File that will play the game. This file includes functions for playing through Othello with human vs human or human vs AI. It additionally includes the function that will load the AI from the saved file.

Iago.py - This file includes functions for loading in data from the original dataset and parsing it into the format that we need it in. It additionally includes functions for taking this parsed data and turning it into the correct format for features for the model. It additionally trains the model, saves it to a file for later use, and evaluates it. The GridSearch is also included in this file, but it is commented out.

WTH_2004.txt - The original dataset we used. Includes data from several professional Othello games from 2004. Games are stored with who won the game and each of the moves made in the game. This file is written in binary.

WTH_dataset_X.txt - File containing the features dataset. If this file does not exist, data is parsed out of WTH_2004 in Iago.py and the file is created. If this file does exist, the data will be loaded when it is needed for training or evaluating.

WTH_dataset_y.txt - File containing the targets dataset. If this file does not exist, data is parsed out of WTH_2004 in Iago.py and the file is created. If this file does exist, the data will be loaded when it is needed for training or evaluating.

saved models (directory) - This folder contains saved, trained models in .h5 format. This file is created in Iago.py and then loaded in main.py in order to make predictions on where the AI should move. Most models are saved with a name indicating their accuracy score (ie, model39.47.h5 has an accuracy score of 39.47%).

Pipfile - Use this file to set up your environment to run the code.

INSTRUCTIONS FOR USE
--------------------

------------------------------------------------------------------------------------------------

SETUP

Install pipenv in order to setup your environment with the following commands:
    pip install pipenv
    pipenv install
    pipenv shell

Next, run the pip file to install dependencies. Note that this code runs on Python 3.6. Please make sure you are running Python 3.6.

------------------------------------------------------------------------------------------------

BUILDING/TRAINING THE MODEL

If you see .h5 models in the saved models directory, you do not need to build a model. At leat one model has already been built and is saved to a file. We do not recommend rebuilding the model as this could take a few hours.

However, if you need to rebuild the model, you need to run main.py (python main.py). When the menu appears, select (2) Train the Model. This file will build the data files if they don't already exist or load them if they do. From here, it will use these datasets to train the model. Once the model is trained, it will be saved to "model[score].h5" where it can be used in the main game.

------------------------------------------------------------------------------------------------

GRIDSEARCH

In the original building of the model, we used GridSearch to find the best parameters for our model. This code still exists in Iago.py, but it is commented out. Since we already found the desired parameters, there is no need to run it again. This is good since the GridSearch took about 28 hours.

------------------------------------------------------------------------------------------------

EVALUATING THE MODEL

Without Training:

If you would like to evaluate a model that has already been trained, run main.py (python main.py) and select option 3 in the menu for evaluating a model. Once this is selected, the dataset will first be loaded so that the test data can be gathered. If the dataset file does not exist yet, it will be created. If it does exist, the data will be loaded. This portion can take a few minutes to run. After the data has been prepared, you will be asked to select a model to evaluate. Type in the full name of the model (including the file extension) to evaluate it. The accuracy score will be printed to the console.

After Training:

Each model is automatically evaluated using the test dataset after it is trained. Once the training completes, the evaluate_model function is called in Iago.py. This will print the model's accuracy score. The accuracy score will then be used as part of the model name in the save file so it can be identified later.

------------------------------------------------------------------------------------------------

PLAYING THE GAME

To play the game, run main.py (python main.py). The game will prompt you for what you want to do. Select the option for playing Human vs. AI. From here, you will be given a list of the saved models. Type the name of the model you wish to play against (including the file extension) and press enter. You will be re-prompted until you enter a valid option. This will bring you to the game interface. The game interface is text based. On your turn, you will be asked to select the column you wish to move to (x value) and the row (y value). If your input is a number that does not correspond to a row or column or if you select a space that you cannot move, you will be re-prompted. 

You play white. Black (the AI) will move first. If one player is forced to skip a turn because they have no available moves, the game will handle this by continuing to prompt the appropriate player until the next player can play again. When the game ends, the winner will be displayed.

To quit in the middle of the game, enter any value that is not a number.

RESOURCES
---------

file format info: http://cassio.free.fr/cassio/custom_install/database/FORMAT_WTHOR.TXT

Code for game logic/console interface: https://github.com/patrickfeltes/othello

File format is the game result followed by the game movelist currently. TODO one hot encoding
game result is 0 for a tie, -1 for a white win, 1 for a black win
