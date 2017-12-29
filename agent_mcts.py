from model.model import Model
import tensorflow as tf
import numpy as np
import random
import  sys
sys.path.append('/home/rick/project/pygo_tt')
from pygo_tt import Env
from copy import deepcopy

class Tree:
    def __init__(self,p,parent,game):
        self.n = 0
        self.w = 0
        self.q = 0
        self.p = p
        self.parent = parent
        self.game = game
        self.child = []
        self.vertex = None  
        

class Agent:
    def __init__(self,conf):
        self.c = conf
        self.model = Model()
        self.sess = tf.Session()
        self.model.load(self.sess)
    def evaluate(self,node):
        board = deepcopy(node.game.env.board)
        board.shape += (1,)
        if node.game.current_color == 'black':
            back = np.ones((9,9,1))
        else:
            back = np.zeros((9,9,1))
         
        state = np.concatenate((board,back),axis=2)

        _r = self.model.inference(self.sess,[state])
        v = _r[0][0]

        p = {}
        for i in range(9):
            for j in range(9):
                p[(i,j)] = _r[1][0][9*i+j]
        p['pass'] = _r[1][0][-1]
        return v, p
    def mcts(self):
        root_node = self.root
        
        curr_node = root_node

        #select
        while len(curr_node.child) > 0:
            u = []
            tmp = []
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
            if curr_node.game.end:
                break

        #expand
        value, policy = self.evaluate(curr_node) 
        if not curr_node.game.end:
            legals = curr_node.game.legal_states()
            for k, v in legals.items():
                p = policy[k]
                g = deepcopy(curr_node.game)
                g.play(k)
                new_child = Tree(p,curr_node,g)
                new_child.vertex = k
                curr_node.child.append(new_child)

        #backup
        while curr_node is not root_node:
            curr_node.n += 1
            curr_node.w += value
            curr_node.q = curr_node.w/curr_node.n
            curr_node = curr_node.parent

    def play(self,game):
        self.root = Tree(1,None,game) 

        sims = 2
        for i in range(sims):
            self.mcts()
        
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
