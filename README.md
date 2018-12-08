# ML-Othello

IAGO: THE OTHELLO AI
====================

FILES
-----

main.py - File that will play the game. This file includes functions for playing through Othello with human vs human or human vs AI. It additionally includes the function that will load the AI from the saved file.

Iago.py - This file includes functions for loading in data from the original dataset and parsing it into the format that we need it in. It additionally includes functions for taking this parsed data and turning it into the correct format for features for the model. It additionally trains the model, saves it to a file for later use, and evaluates it. The GridSearch is also included in this file, but it is commented out.

WTH_2004.txt - The original dataset we used. Includes data from several professional Othello games from 2004. Games are stored with who won the game and each of the moves made in the game. This file is written in binary.

WTH_dataset_X.txt - File containing the features dataset. If this file does not exist, data is parsed out of WTH_2004 in Iago.py and the file is created. If this file does exist, the data can simply be loaded in Iago.py next time it is run.

WTH_dataset_y.txt - File containing the targets dataset. If this file does not exist, data is parsed out of WTH_2004 in Iago.py and the file is created. If this file does exist, the data can simply be loaded in Iago.py next time it is run.

model.h5 - The saved, trained model. This file is created in Iago.py and then loaded in main.py in order to make predictions on where the AI should move.

INSTRUCTIONS FOR USE
--------------------

// TODO: Write this section

How to use this program:
install pipenv inorder to setup your environment with the following commands:
    pip install pipenv
    pipenv install
    pipenv shell
Now run your code

File format is the game result followed by the game movelist currently. TODO one hot encoding
game result is 0 for a tie, -1 for a white win, 1 for a black win

HOW IT WORKS
------------

// TODO: Write this section

RESOURCES
---------

file format info: http://ledpup.blogspot.com/2012/03/computer-reversi-part-15-thor-database.html

Code for game logic/console interface: https://github.com/patrickfeltes/othello