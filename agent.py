from model.model import Model
import tensorflow as tf
import numpy as np
import random

class Agent:
    def __init__(self,eps):
        self.eps = eps
        self.model = Model()
        self.sess = tf.Session()
        self.model.load(self.sess)
    def evaluate(self,batch):
        _result = self.model.inference(self.sess,batch)
        return _result
    def play(self,color,states):
#        _b_cap = np.array([s[1][1] for s in states])
#        _b_cap.shape += (1,)
#        _w_cap = np.array([s[1][2] for s in states])
#        _w_cap.shape += (1,)

#        batch = [_states,_b_cap,_w_cap]

        if random.random() < self.eps:
            return random.choice(states.keys())
        else:
            _vertex = []
            _states = []

            for k, v in states.items():
                _vertex.append(k)
                _states.append(v)

            _states = np.array(_states)
            _states.shape += (1,)

            batch = [_states]

            _result = self.evaluate(batch)

            if color == 'black':
                idx = np.argmax(_result)
            else:
                idx = np.argmin(_result)

            return _vertex[idx]

