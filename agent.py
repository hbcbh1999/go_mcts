from model.model import Model
import tensorflow as tf
import numpy as np


class Agent:
    def __init__(self):
        self.model = Model()
        self.sess = tf.Session()
        self.model.load(self.sess)
    def evaluate(self,batch):
        _result = self.model.inference(self.sess,batch)
        return _result
    def play(self,color,states):
        _states = np.array([s[1][0] for s in states])
        _states.shape += (1,)
        _b_cap = np.array([s[1][1] for s in states])
        _b_cap.shape += (1,)
        _w_cap = np.array([s[1][2] for s in states])
        _w_cap.shape += (1,)

        batch = [_states,_b_cap,_w_cap]

        _result = self.evaluate(batch)

        if color == 'black':
            idx = np.argmax(_result)
        else:
            idx = np.argmin(_result)
        return states[idx][0]
    def save(self):
        self.model.save(self.sess)
