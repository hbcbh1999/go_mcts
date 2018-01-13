import sys
from os.path import dirname
sys.path.append(dirname(__file__)+'/../../pygo_tt')
from pygo_tt import Env
import os.path
import pickle
import numpy as np
def swap_color(color):
    if color == 'black':
        return 'white'
    else:
        return 'black'

class GoGame:

    def __init__(self,boardsize,clone=False):
        self.boardsize = boardsize
        self.env = Env(boardsize=boardsize)
        self.reset()

    def reset(self):
        self.env.reset()
        self.current_color = 'black'
        self.last_vertex = ''
        self.end = False
        self.winner = None
        self.n_plays = 0
    def play(self,vertex):
        self.n_plays += 1
        if (vertex == self.last_vertex and vertex == 'pass') or vertex == 'resign':
            self.force_end()
        elif vertex != 'pass':
            self.env.play(self.current_color,vertex)
        if self.n_plays > 1.5 * self.boardsize ** 2:
            self.force_end()
        self.last_vertex = vertex
        self.current_color = swap_color(self.current_color)
    def force_end(self):
        self.end = True
        score = self.env.score()
        if score > 0:
            self.winner = 'black'
        elif score < 0:
            self.winner = 'white'
        else:
            self.winner = 'draw'

    def print_board(self):
        sys.stdout.write(self.env.board)
    def legal_states(self):
        return self.env.legals[self.current_color]
    def save(self,file_name):
        pass
        #_to_save = [self.winner, self.history]
        #with open(file_name,'w') as f:
        #    pickle.dump(_to_save,f)
    def clone(self):
        tmp = GoGame.__new__(GoGame)
        for k, v in list(self.__dict__.items()):
            if k is 'env':
                env_tmp = Env.__new__(Env)
                env_tmp.neighbors = self.env.neighbors
                env_tmp.boardsize = self.env.boardsize
                env_tmp.history_hash = self.env.history_hash.copy()
                env_tmp.liberty = np.copy(self.env.liberty)
                env_tmp.legals = self.env.legals.copy()
                env_tmp.board = np.copy(self.env.board)
                env_tmp.komi = self.env.komi
                env_tmp.history = [x for x in self.env.history]
                env_tmp.legal_boards = self.env.legal_boards.copy()
                tmp.env = env_tmp
            elif k is 'history':
                tmp.history = self.env.history.copy()
            else:
                setattr(tmp,k,v)
        return tmp
