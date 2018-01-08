from model.model import Model
import tensorflow as tf
import numpy as np
import random
import  sys
sys.path.append('/home/rick/project/pygo_tt')
from pygo_tt import Env
from copy import deepcopy
import pickle

class Tree:
    def __init__(self,parent,idx=None,vertex=None):
        self.parent = parent
        self.child = []
        #n,w,p,q
        self.child_stats = None
        self.vertex = vertex  
        self.visited = False
        self.game = None
        self.idx = idx
    def copy_and_play(self):
#        self.game = deepcopy(self.parent.game)
        self.game = self.parent.game.clone()
        self.game.play(self.vertex)
        self.visited = True

class Agent:
    def __init__(self,conf,sims):
        self.c = conf
        self.sims = sims
        self.model = Model()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.model.load(self.sess)
    def evaluate(self,node):
        board = np.copy(node.game.env.board)
        board.shape += (1,)
        if node.game.current_color == 'black':
            back = np.ones((9,9,1))
        else:
            back = np.zeros((9,9,1))
         
        state = np.concatenate((board,back),axis=2)

        _r = self.model.inference(self.sess,[state])
        v = _r[0][0][0]

        p = {}
        for i in range(9):
            for j in range(9):
                p[(i,j)] = _r[1][0][9*i+j]
        p['pass'] = _r[1][0][-1]
        return v, p
    def mcts(self,root_node):
        
        curr_node = root_node

        #select
        while curr_node.child:
            n_child = len(curr_node.child)

            #n,p,q
            _stats = curr_node.child_stats

            _v = _stats[3] + self.c * np.sqrt(np.sum(_stats[0])) * _stats[2] / ( 1 + _stats[0] ) 

            max_i = np.argmax(_v)

            curr_node = curr_node.child[max_i]

        #expand
        if not curr_node.visited:
            curr_node.copy_and_play()
        if not curr_node.game.end:

            value, policy = self.evaluate(curr_node)
            legals = curr_node.game.legal_states()

            curr_node.child_stats = np.zeros((4,len(legals)))
            _stats = curr_node.child_stats

            for i, (k, v) in enumerate(legals.items()):
                _stats[2][i] = policy[k]
                _stats[3][i] = 1
                new_child = Tree(curr_node,idx=i,vertex=k)
                curr_node.child.append(new_child)
        else:
            if curr_node.game.winner == curr_node.game.current_color:
                value = 1.
            else:
                value = 0.
        #backup
        color = curr_node.game.current_color 
        while curr_node is not root_node:
            parent = curr_node.parent
            curr_idx = curr_node.idx
            parent.child_stats[0][curr_idx] += 1
            if curr_node.game.current_color == color:
                parent.child_stats[1][curr_idx] += (1-value)
            else:
                parent.child_stats[1][curr_idx] += value
            parent.child_stats[3][curr_idx] = parent.child_stats[1][curr_idx] / parent.child_stats[0][curr_idx]
            curr_node = parent

    def play(self):

        for i in range(self.sims):
            self.mcts(self.root)
        
        temp = 1.

        n_temp = self.root.child_stats[0] ** (1/temp)
        self.prob = n_temp / np.sum(n_temp)

        c_idx = np.random.choice(len(self.prob),p=self.prob)
        
        c_v = self.root.child[c_idx].vertex

        return c_v

    def get_state(self):
        if self.root.game.current_color == 'black':
            back = np.ones((9,9,1))
        else:
            back = np.zeros((9,9,1))
        return np.concatenate((self.root.game.env.board,back),axis=2)

    def get_prob(self):
        p_map = np.zeros(9*9+1)
        for i, p in enumerate(self.prob):
            _v = self.root.child[i].vertex
            if _v == 'pass':
                p_map[-1] = p
            else:
                idx = _v[0] * 9 + _v[1]
                p_map[idx] = p
        return p_map
    def set_root(self,game):
        self.root = Tree(1,None)
#        self.root.game = deepcopy(game)
        self.root.game = game.clone()
        self.root.visited = True
    def update_root(self,vertex):
        for c in self.root.child:
            if c.vertex == vertex:
                self.root = c
                if not c.visited:
                    c.copy_and_play()
                c.parent = None
                break
