import sys
sys.path.append('/home/rick/project/pygo')
import pygo
import os.path
import pickle

def swap_color(color):
    if color == 'black':
        return 'white'
    else:
        return 'black'

class GoGame:

    def __init__(self,boardsize):
        self.env = pygo.Env()
        self.env.boardsize(boardsize)
        self.reset() 
    def reset(self):
        self.env.clear_board()
        self.current_color = 'black'
        self.last_vertex = ''
        self.end = False
        self.winner = None
        self.history = []
    def play(self,vertex):
        if vertex != 'pass':
            if vertex == 'genmove':
                self.env.genmove(self.current_color)
            else:
                self.env.play(self.current_color,vertex)
            self.history.append(self.env.get_state())
        elif (vertex == self.last_vertex and vertex == 'pass') or vertex == 'resign':
            self.end = True
#            print self.env.final_score()
            self.winner = self.env.get_winner()
        self.last_vertex = vertex
        self.current_color = swap_color(self.current_color)
    def print_board(self):
        print self.env.showboard()
    def legal_states(self):
        return self.env.try_all_legal(self.current_color)
    def save(self,file_name):
        _to_save = [self.winner, self.history]
        with open(file_name,'w') as f:
            pickle.dump(_to_save,f)
