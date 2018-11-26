# ML-Othello

file format info: http://ledpup.blogspot.com/2012/03/computer-reversi-part-15-thor-database.html

Code for game logic/console interface: https://github.com/patrickfeltes/othello

How to use this program:
install pipenv inorder to setup your environment with the following commands:
    pip install pipenv
    pipenv install
    pipenv shell
Now run your code

File format is the game result followed by the game movelist currently. TODO one hot encoding
game result is 0 for a tie, -1 for a white win, 1 for a black win