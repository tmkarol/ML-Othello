import math

def PrintBoard(board):
    '''
    Function to print the board's current state, plus the current score.
    '''
    print("  0 1 2 3 4 5 6 7 ")
    for r in range(len(board)):
        s = str(r) + '|'
        for c in range(len(board[r])):
            s += board[r][c] + '|'
        print(s + str(r))
    print("  0 1 2 3 4 5 6 7 ")
    (black, white) = GetScore(board)
    print("SCORE:")
    print('B: ' + str(black))
    print('W: ' + str(white))

def GetScore(board):
    '''
    Function returns both players' current scores.
    '''
    black = 0
    white = 0
    
    for r in board:
        for c in r:
            if c == 'B': 
                black += 1
            elif c == 'W':
                white += 1
    return (black, white)

def GetPossibleMoves(board, player):
    '''
    Returns a list of all possible moves for a given player.
    '''
    moves = []

    for x in range(len(board)):
        for y in range(len(board)):
            if not IsLegalMove(board, y, x, player): 
                continue
            else:
                if len(GetPiecesToFlip(board, x, y, player)) > 0:
                    moves.append((x, y))
    return moves

def IsLegalMove(board, r, c, player):
    '''
    Checks if a space is empty.
    '''
    return board[r][c] == ' '

def GetIncludedPieces(board, x_start, y_start, x_dir, y_dir, player):
    '''
    Get all pieces that should be flipped given a move and a direction.
    '''
    included = []

    if player == 'B':
        other_player = 'W'
    else:
        other_player = 'B'

    # distance is 7 spaces
    for dist in range(1, 8):
        x_curr = x_start + dist * x_dir
        y_curr = y_start + dist * y_dir

        # if the current position is off the board, return [] because the pieces are not bounded
        if x_curr < 0 or x_curr >= len(board) or y_curr < 0 or y_curr >= len(board):
            return []

        if board[y_curr][x_curr] == other_player:
            included.append((x_curr, y_curr))
        elif board[y_curr][x_curr] == player:
            return included
        else:
            return []

    return []

def GetPiecesToFlip(board, x, y, player):
    '''
    Gat all pieces that should be flipped from a given move.
    '''
    # get positions of all pieces to be flipped by a move
    flip = []

    # all different directions, xDir = 0 and yDir = 0 not included because then there wouldn't be a direction!
    flip.extend((GetIncludedPieces(board, x, y, 1, 1, player)))
    flip.extend((GetIncludedPieces(board, x, y, 1, -1, player)))
    flip.extend((GetIncludedPieces(board, x, y, -1, 1, player)))
    flip.extend((GetIncludedPieces(board, x, y, 0, 1, player)))
    flip.extend((GetIncludedPieces(board, x, y, 0, -1, player)))
    flip.extend((GetIncludedPieces(board, x, y, 1, 0, player)))
    flip.extend((GetIncludedPieces(board, x, y, -1, 0, player)))
    flip.extend((GetIncludedPieces(board, x, y, -1, -1, player)))

    # use a set to remove duplicates
    return list(set(flip))

def FlipPieces(board, flip, player):
    '''
    Flip all pices that should be flipped.
    Flipped is the list of pieces to be flipped.
    Player is the player that is getting the pieces.
    '''
    for pos in flip:
        board[pos[1]][pos[0]] = player

    return board

def PromptMove(board, player):
    '''
    Asks the (human) player to make a move.
    This function will loop until it receives a valid input.
    '''
    print(player + " player's turn!")
        
    possibilites = GetPossibleMoves(board, player)

    # move can't be made! let other player move!
    if len(possibilites) == 0:
        return False

    x_move = -1
    y_move = -1

    while (x_move, y_move) not in possibilites:
        while x_move < 0 or x_move >= len(board):
            x_move = int(input("Enter a x coordinate(column): "))

        while y_move < 0 or y_move >= len(board):
            y_move = int(input("Enter a y coordinate(row): "))

        if (x_move, y_move) not in possibilites:
            x_move = -1
            y_move = -1

    flip = GetPiecesToFlip(board, x_move, y_move, player)
    board[y_move][x_move] = player

    board = FlipPieces(board, flip, player)

    return board

def IsBoardFull(board):
    '''
    Checks to see if the board is full (and the game is complete).
    '''
    full = True

    for r in board:
        for c in r:
            if c == ' ':
                full = False
    return full

def RunNoAI():
    '''
    Sets up and runs the game for human vs. human.
    '''
    # create 8 by 8 board
    board = []
    for x in range(8):
        board.append([' '] * 8)
    
    board[3][3] = 'W'
    board[3][4] = 'B'
    board[4][3] = 'B'
    board[4][4] = 'W'

    player = 'B'
    other_player = 'W'

    while not IsBoardFull(board):
        PrintBoard(board)

        # game over!
        if len(GetPossibleMoves(board, player)) == 0 and len(GetPossibleMoves(board, other_player)) == 0:
            break

        tmp = PromptMove(board, player)
        if not tmp == False:
            board = tmp
            
        (player, other_player) = (other_player, player)    

    (black, white) = GetScore(board)

    if black > white:
        print("Black wins!")
    elif black < white:
        print("White wins!")
    else:
        print("Tie?")

def RunOneAI():
    '''
    Run a game with human vs. AI
    For demo purposes
    '''
    # create 8 by 8 board
    board = []
    for x in range(8):
        board.append([' '] * 8)
    
    board[3][3] = 'W'
    board[3][4] = 'B'
    board[4][3] = 'B'
    board[4][4] = 'W'

    player = 'B' # Human
    other_player = 'W' # AI

    while not IsBoardFull(board):
        PrintBoard(board)

        # game over!
        if len(GetPossibleMoves(board, player)) == 0 and len(GetPossibleMoves(board, other_player)) == 0:
            break

        if player == 'B':
            tmp = PromptMove(board, player)
        else:
            # TODO: Make AI decsision
            # TODO: Call MakeAIMove
            raise NotImplementedError
        if not tmp == False:
            board = tmp
            
        (player, other_player) = (other_player, player)    

    (black, white) = GetScore(board)

    if black > white:
        print("Black wins!")
    elif black < white:
        print("White wins!")
    else:
        print("Tie?")

def RunTwoAI():
    '''
    Run a game with AI vs. AI
    For training/testing purposes
    '''
    # create 8 by 8 board
    board = []
    for x in range(8):
        board.append([' '] * 8)
    
    board[3][3] = 'W'
    board[3][4] = 'B'
    board[4][3] = 'B'
    board[4][4] = 'W'

    player = 'B' 
    other_player = 'W' 

    while not IsBoardFull(board):
        PrintBoard(board)

        # game over!
        if len(GetPossibleMoves(board, player)) == 0 and len(GetPossibleMoves(board, other_player)) == 0:
            break

        # TODO: Make AI decsision
        # TODO: Call MakeAIMove
        raise NotImplementedError
        if not tmp == False:
            board = tmp
            
        (player, other_player) = (other_player, player)    

    (black, white) = GetScore(board)

    if black > white:
        print("Black wins!")
    elif black < white:
        print("White wins!")
    else:
        print("Tie?")

def MakeAIMove(board, player, move):
    '''
    Function takes in the decided move from the AI and makes that move
    AI keeps the moves as column + (row * 10)
    '''

    # Convert given row and column to 0-7 rows and columns
    x_move = (move % 10) - 1 # Column
    y_move = math.floor(move / 10) - 1 # Row

    # Make the move
    flip = GetPiecesToFlip(board, x_move, y_move, player)
    board[y_move][x_move] = player

    board = FlipPieces(board, flip, player)

    return board

def PromptGameType():
    '''
    Asks the user how many AI they'd like to use, running the appropriate game afterward
    '''
    print("Welcome to OTHELLO AI!")
    print("Type the number for the type of game you'd like to run.")
    print("0: Human vs. Human")
    print("1: Human vs. AI")
    print("2: AI vs. AI")
    choice = -1
    while choice != 1:
        choice = int(input("Your choice: "))
        if choice == 0:
            RunNoAI()
        elif choice == 1:
            RunOneAI()
        elif choice == 2:
            RunTwoAI()
        else:
            print("Please enter a valid choice.")
            choice = -1
    print("\n") #???

# Run the game!
#PromptGameType()
