


import numpy as np

BOARD_DIM = 3
gamma = 0.9
eps = 0.1
TRAINING_RUNS = 10000



class Board():
    
    def __init__(self):
        #initialize the board with all zeros and then players fill in 1 and -1
        self.square = np.zeros([BOARD_DIM,BOARD_DIM], dtype='int')
        self.x = -1
        self.o = 1
        
    def board_permutation_number(self):
        e = 0 #initial exponent is zero
        bpn = 0 #board permutation number
        for i in range(BOARD_DIM):
            for j in range(BOARD_DIM):
                bpn += self.get_place_value(i,j)*(BOARD_DIM**e)
                #print("ind place values:{}".format(bpn))
                e += 1
        return bpn
       
    
    def list_empty_squares(self):
        empty_squares = []
        
        for i in range(BOARD_DIM):
            for j in range(BOARD_DIM):
                if self.square[i,j] == 0:
                    empty_squares.append((i,j))   
        return empty_squares
        
    def make_move(self, player, i, j):
        if player==1 and self.square[i,j] == 0:
            self.square[i,j] = self.x 
        elif player==2 and self.square[i,j] == 0:
            self.square[i,j] = self.o
        else:
            print("Wrong move ! Please re-enter")
            return
        
    def reverse_move(self, i, j):
        self.square[i,j] = 0
     
    # this function returns whether the square contains an x or o or is empty
    def get_place_value(self, i, j):
        if self.square[i,j] == self.x:
            return 1
        elif self.square[i,j] == self.o:
            return 2
        else: return 0
        
    def check_win_draw(self):
        for i in range(BOARD_DIM):
            if np.sum(self.square[:,i]) == -3:
                return self.x
            if np.sum(self.square[:,i]) == 3:
                return self.o 
            if np.sum(self.square[i,:]) == -3:
                return self.x
            if np.sum(self.square[i,:]) == 3:
                return self.o 
            
        if np.trace(self.square) == -3:
            return self.x
        elif np.trace(self.square) == 3:
            return self.o 
        elif np.trace(np.fliplr(self.square)) == -3:
            return self.x 
        elif np.trace(np.fliplr(self.square)) == 3:
            return self.o
        elif len(self.list_empty_squares())==0:            
            return 0.5 # its a draw !!
        return 0 # not a win or draw keep playing
    
    def draw_board(self):
        current_board = self.square
        current_board = np.where(self.square==0 , '*' , current_board)
        current_board = np.where(self.square==-1 , 'x' , current_board)
        current_board = np.where(self.square==1 , 'o' , current_board)
        print(f"{current_board}")
        
    def reset(self):
        for i in range(BOARD_DIM):
            for j in range(BOARD_DIM):
                self.square[i,j] = 0
                
    def reset_single_square(self, i, j):
        self.square[i,j] = 0 

    #accept move from player and parse it.
    def grab_move_entered(self, inputstring):
        
        numintsgrabbed = 0
        for element in inputstring:
            if (element.isdigit() and numintsgrabbed == 0):
                i = element
                numintsgrabbed += 1
            elif (element.isdigit() and numintsgrabbed == 1):
                j = element
                return self.check_valid_move(i, j), i, j
        return 0, 0, 0
    
    #check if player has made a valid move    
    def check_valid_move(self, i, j):

        if int(i) not in [0,1,2] or int(j) not in [0,1,2]:
            print('That was not a valid move.\n')
            return 0
        if self.square[int(i),int(j)] != 0:
            print('That move was already made.\n')
            return 0
        return 1
        



class Agent():
    
# first player plays with 'x's and auto=False means the player is a human    
    def __init__(self, player_num, auto=True):
        self.player_num = player_num
        self.auto = auto
        self.state_value = {}
        self.current_game = []
        #initialise the result to a draw and change the value if win or loss
        self.result = 0.5
        
        
    def make_move(self, opponent, board):
        validmove = 0
        if self.auto == False:
            while validmove == 0:
                inputstring = input("Enter your move: row (0-2) col (0-2):")  
                validmove, i, j = board.grab_move_entered(inputstring)
            board.make_move(self.player_num, int(i), int(j))
            self.current_game.append((int(i), int(j)))
            opponent.current_game.append((int(i), int(j)))
        else:
            es = board.list_empty_squares()
            if np.random.random() < eps:
                random_square = np.random.randint(len(es))
                best_i = es[random_square][0]
                best_j = es[random_square][1]
                
            else:
                max_v = 0
                best_i = es[0][0]
                best_j = es[0][1]

                #lets assume no randomness for now
                for i, j in es:
                    board.make_move(self.player_num, int(i), int(j))
                    bpn = board.board_permutation_number()
                    self.add_state(bpn, opponent, board)
                    new_v = self.state_value[bpn][0]

                    if(new_v > max_v):
                        best_i=int(i)
                        best_j=int(j)
                        max_v = new_v

                    board.reverse_move(int(i),int(j)) 
                
            board.make_move(self.player_num, best_i, best_j)
            self.current_game.append((best_i, best_j))
            opponent.current_game.append((best_i, best_j))
            
            
    #take in the board permutation number and update the state values for the agent       
    def add_state(self, bpn, opponent, board):
        if bpn not in self.state_value:
            is_win = board.check_win_draw()
            if (is_win == -1 or is_win == 1):
                #if there is a win then naturally the self is the agent that won  
                # and the opponent is the agent that lost
                self.state_value.update({bpn:[1,1]})
                opponent.state_value.update({bpn:[0,1]})     
            else:
                self.state_value.update({bpn:[0.5,1]})
                opponent.state_value.update({bpn:[0.5,1]})
                

    # update the state_value dictionary for both players based upon the current_game and result
    def update_game_state_values(self, opponent, board):
        # get the reward values that we need to percolate down the states reached during the game
        reward = self.result
        opponent_reward = opponent.result
        
        while len(self.current_game) > 1:
            i, j = self.current_game.pop()  #pop the final move whose value will always be constant
            board.reset_single_square(i,j)  #get board state before the final move

            bpn = board.board_permutation_number()
            reward *= gamma
            opponent_reward *= gamma
            if bpn in self.state_value.keys():
                self.update_single_state_value(opponent, bpn, reward, opponent_reward)
        
            
    # takes the latest state value and averages out with the current value of that state to get the
    # current averaged out state value
    def update_single_state_value(self, opponent, bpn, reward, opponent_reward):
        current_state_value = self.state_value[bpn][0]
        times_state_reached = self.state_value[bpn][1]
        
        new_state_value = ((current_state_value*times_state_reached)+reward)/(times_state_reached+1)
        
        self.state_value.update({bpn:[new_state_value, times_state_reached+1]})
        
        opp_current_state_value = opponent.state_value[bpn][0]
        opp_times_state_reached = opponent.state_value[bpn][1]
        
        opp_new_state_value = ((opp_current_state_value*opp_times_state_reached)+opponent_reward)/(opp_times_state_reached+1)
        
        opponent.state_value.update({bpn:[opp_new_state_value, opp_times_state_reached+1]})
        
        


def train(b, p1, p2):
   
   print("Hi I use reinforcement learning to play tictactoe so you just can't beat me :) ")
   print("First let me play 10000 games against myself and use RL to learn the game:\n")
   for i in range(TRAINING_RUNS):
       if(i%500 == 0): print(i) 
       while True:
           p1.make_move(p2, b)
           #b.draw_board()
           if(b.check_win_draw() == b.x):
               p1.result = 1
               p2.result = 0
               #print('I win !!')
               break
           elif (b.check_win_draw() == 0.5): 
               # last move will always be made by player 1 in a drawn game
               #print('Its a draw !! You're pretty good.')
               break

           p2.make_move(p1, b)
           #b.draw_board()
           if(b.check_win_draw() == b.o):
               p1.result = 0
               p2.result = 1
               #print('You win !!')
               break
           
       p1.update_game_state_values(p2, b)
       #p1.update_game_state_values(b)
       b.reset()
       
   
   
def play_game(b, p1):
   
   global eps
   while True:
       
       eps = 0  #setting epsilon = 0 EXPLOIT MODE
       play = input("Do you want to play (y/n)")
       if play == 'y':
           b.reset()
           p3 = Agent(player_num=2, auto=False)
           while True:
               p1.make_move(p3, b)
               b.draw_board()
               if(b.check_win_draw() == b.x):
                   p1.result = 1
                   p3.result = 0
                   print('I win !! Brush up your game buddy.')
                   break
               elif (b.check_win_draw() == 0.5): 
                   # last move will always be made by player 1 in a drawn game
                   print('Its a draw !! You are pretty good.')
                   break

               p3.make_move(p1, b)
               #b.draw_board()
               if(b.check_win_draw() == b.o):
                   p1.result = 0
                   p3.result = 1
                   print('You win !! Whatta playa.')
                   break
                   
       elif play == 'n':
           print("Hope you enjoyed the game !")
           break





#initialise the game board and the agent
gameboard = Board()
player1 = Agent(player_num=1)
player2 = Agent(player_num=2)

#train the agent by playing itself over 10000 games
train(gameboard, player1, player2)

#agent plays with a human
play_game(gameboard, player1)






