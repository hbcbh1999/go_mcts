import sys
sys.path.append('/home/rick/project/pygo_tt')
from pygo_tt import Env
import os.path
import pickle

def swap_color(color):
    if color == 'black':
        return 'white'
    else:
        return 'black'

class GoGame:

    def __init__(self,boardsize):
        self.env = Env(boardsize=boardsize)
        self.reset() 
    def reset(self):
        self.env.reset()
        self.current_color = 'black'
        self.last_vertex = ''
        self.end = False
        self.winner = None
        self.history = []
    def play(self,vertex):
        if (vertex == self.last_vertex and vertex == 'pass') or vertex == 'resign':
            self.end = True
#            print self.env.final_score()
            score = self.env.score()
            if score > 0:
                self.winner = 'black'
            elif score < 0:
                self.winner = 'white'
            else:
                self.winner = 'draw'
        elif vertex != 'pass':
            self.env.play(self.current_color,vertex)
        self.history.append(self.env.board)
        self.last_vertex = vertex
        self.current_color = swap_color(self.current_color)
    def print_board(self):
        print self.env.board
    def legal_states(self):
        return self.env.legals[self.current_color]
    def save(self,file_name):
        _to_save = [self.winner, self.history]
        with open(file_name,'w') as f:
            pickle.dump(_to_save,f)
