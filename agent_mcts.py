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
    def __init__(self,p,parent):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = p
        self.parent = parent
        self.child = []
        self.vertex = None  
        self.visited = False
        self.game = None
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
        while len(curr_node.child) > 0:
            u = []
            n_sum = 0
            for c in curr_node.child:
                n_sum += c.n
                u.append(self.c*c.p/(1+c.n))
            max_i = 0
            max_val = -1
            for i in range(len(u)):
                u[i] *= np.sqrt(n_sum)
                val = u[i]+curr_node.child[i].q
                if val > max_val:
                    max_val = val
                    max_i = i
            curr_node = curr_node.child[i]

        #expand
        if not curr_node.visited:
            curr_node.copy_and_play()
        if not curr_node.game.end:
            value, policy = self.evaluate(curr_node)
            legals = curr_node.game.legal_states()
            for k, v in legals.items():
                p = policy[k]
                new_child = Tree(p,curr_node)
                new_child.vertex = k
                curr_node.child.append(new_child)
        else:
            if curr_node.game.winner == curr_node.game.current_color:
                value = 0.
            else:
                value = 1.
        #backup
        color = curr_node.game.current_color 
        while curr_node is not root_node:
            curr_node.n += 1
            if curr_node.game.current_color == color:
                curr_node.w += (1-value)
            else:
                curr_node.w += value
            curr_node.q = curr_node.w/curr_node.n
            curr_node = curr_node.parent

    def play(self):

        for i in range(self.sims):
            self.mcts(self.root)
        
        temp = 1.
        n_s = np.zeros((9*9+1,))
        for c in self.root.child:
            if c.vertex == 'pass':
                n_s[-1] = c.n**(1/temp)
            else:
                n_s[9*c.vertex[0]+c.vertex[1]] = c.n**(1/temp)
        self.p = n_s/np.sum(n_s)
        a_s = np.arange(len(n_s))
        idx = np.random.choice(np.arange(82),p=self.p)
        if idx == 81:
            return 'pass'
        else:
            return (idx//9,idx%9)

    def get_state(self):
        if self.root.game.current_color == 'black':
            back = np.ones((9,9,1))
        else:
            back = np.zeros((9,9,1))
        return np.concatenate((self.root.game.env.board,back),axis=2)

    def get_prob(self):
        return self.p
    def set_root(self,game):
        self.root = Tree(1,None)
#        self.root.game = deepcopy(game)
        self.root.game = game.clone()
        self.root.visited = True
    def update_root(self,vertex):
        for c in self.root.child:
            if c.vertex == vertex:
                self.root = c
                break
