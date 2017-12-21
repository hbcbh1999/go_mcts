from model.model import Model
import tensorflow as tf
import numpy as np
import pickle 
import os
import sys
import glob
import argparse
import random
"""
ARGS
"""
parser = argparse.ArgumentParser()
parser.add_argument('--new',default=False,help='Create a new model instead of training the old one',action='store_true')
parser.add_argument('--batch_size',default=32,type=int,help='Batch size (default:32)')
parser.add_argument('--epochs',default=10,type=int,help='Training epochs (default:10)')
parser.add_argument('--data_dir',default=['./data'],nargs='*',help='Training data directories (default:./data/ep*)')
parser.add_argument('--n_episodes',default=-1,type=int,help='Number of training episodes (randomly chosen)')
args = parser.parse_args()

new = args.new
batch_size = args.batch_size
epochs = args.epochs
data_dir = args.data_dir
n_episodes = args.n_episodes

""" LOAD DATA """

list_of_data = []
for _d in data_dir:
    list_of_data += glob.glob(_d+'/ep*') 

if n_episodes == -1 or n_episodes > len(list_of_data):
    _list_eps = list_of_data
elif n_episodes < len(list_of_data):
    _list_eps = random.sample(list_of_data,n_episodes)

episodes = []
for file_name in list_of_data:
    with open(file_name,'r') as f:
        episodes.append(pickle.load(f))

_states = []
_b_cap = []
_w_cap = []
_label = []

for ep in episodes:
    if ep[0] == 'black':
        _label += [1.]*len(ep[1])
    elif ep[0] == 'white':
        _label += [0.]*len(ep[1])
    else:
        _label += [0.5]*len(ep[1])
#    _tmp = zip(*ep[1])
#    _states += list(_tmp[0])
#    _b_cap += list(_tmp[1])
#    _w_cap += list(_tmp[2])
    _states += ep[1]
_states = np.array(_states)

''' ROTATION AUG '''
#_states = np.concatenate([_states,np.rot90(_states,1,(1,2)),np.rot90(_states,2,(1,2)),np.rot90(_states,3,(1,2))])

''' FLIP AUG '''
#_states = np.concatenate([_states,np.flip(_states,1),np.flip(_states,2),np.flip(np.flip(_states,1),2)])
_states.shape += (1,)

''' REPEAT '''
multiplicity = 1
_b_cap = np.repeat(np.array(_b_cap),multiplicity)
_b_cap.shape += (1,)
_w_cap = np.repeat(np.array(_w_cap),multiplicity)
_w_cap.shape += (1,)
_label = np.repeat(np.array(_label),multiplicity)
_label.shape += (1,)


"""========================="""

n_data = len(_states)

#iters = min(epochs*n_data/batch_size,100000)
iters = epochs*n_data/batch_size

loss_ma = 0
decay = 0.99

with tf.Session() as sess:
    m = Model()
    if new:
        m.build_graph()
        sess.run(tf.global_variables_initializer())
    else:
        m.load(sess)
    
    for i in xrange(iters):

        idx = np.random.randint(len(_states),size=batch_size)
        #batch = [_states[idx],_b_cap[idx],_w_cap[idx],_label[idx]]
        batch = [_states[idx],_label[idx]]
        
        loss = m.train(sess,batch,i)

        loss_ma = decay * loss_ma + ( 1 - decay ) * loss

        print '\riter:%d/%d loss: %.5f'%(i,iters,loss_ma),
        sys.stdout.flush()
    
    m.save(sess)

