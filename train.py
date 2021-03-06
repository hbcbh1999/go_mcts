from model.model import Model
import tensorflow as tf
import numpy as np
import pickle 
import os
import sys
import glob
import argparse
import random
import pandas as pd
"""
ARGS
"""
parser = argparse.ArgumentParser()
parser.add_argument('--cycle',default=-1,type=int,help='Cycle (default:-1)')
parser.add_argument('--new',default=False,help='Create a new model instead of training the old one',action='store_true')
parser.add_argument('--batch_size',default=32,type=int,help='Batch size (default:32)')
parser.add_argument('--epochs',default=10,type=int,help='Training epochs (default:10)')
parser.add_argument('--max_iters',default=100000,type=int,help='Max training iterations (default:100000)')
parser.add_argument('--data_dir',default=['./data'],nargs='*',help='Training data directories (default:./data/ep*)')
parser.add_argument('--n_episodes',default=-1,type=int,help='Number of training episodes (randomly chosen)')
parser.add_argument('--save_loss',default=False,help='Save loss history',action='store_true')
parser.add_argument('--save_interval',default=100,type=int,help='Number of iterations between save_loss')
args = parser.parse_args()

cycle = args.cycle
new = args.new
batch_size = args.batch_size
epochs = args.epochs
data_dir = args.data_dir
n_episodes = args.n_episodes
max_iters = args.max_iters
save_loss = args.save_loss
save_interval = args.save_interval
""" LOAD DATA """

list_of_data = []
for _d in data_dir:
    list_of_data += glob.glob(_d+'/data*') 

dfs = []
for file_name in list_of_data:
    df = pd.read_pickle(file_name)
    dfs.append(df)
    
df = pd.concat(dfs,ignore_index=True)

b_shape = df['board'][0].shape

white_back = np.zeros(b_shape)
black_back = np.ones(b_shape)

back_map = {'black':black_back,'white':white_back}

back = df['color'].map(back_map)
backs = np.stack(back.values)

boards = np.stack(df['board'].values)
states = np.stack([boards,backs],axis=-1)
policy = np.stack(df['policy'].values)
labels = np.stack(df['result'].values)[:,None]

"""
''' ROTATION AUG '''
#_states = np.concatenate([_states,np.rot90(_states,1,(1,2)),np.rot90(_states,2,(1,2)),np.rot90(_states,3,(1,2))])

''' FLIP AUG '''
#_states = np.concatenate([_states,np.flip(_states,1),np.flip(_states,2),np.flip(np.flip(_states,1),2)])
_states.shape += (1,)

''' REPEAT '''
multiplicity = 1
_label = np.repeat(np.array(_label),multiplicity)
_label.shape += (1,)


"""#=========================


n_data = len(states)

iters = int(min(epochs*n_data//batch_size,max_iters))

loss_ma = 0
decay = 0.99

if save_loss:
    loss_history = []

with tf.Session() as sess:
    m = Model()
    if new:
        m.build_graph()
        sess.run(tf.global_variables_initializer())
    else:
        m.load(sess)
    
    for i in range(iters):

        idx = np.random.randint(n_data,size=batch_size)
        batch = [states[idx],labels[idx],policy[idx]]
        
        loss = m.train(sess,batch,i)

        loss_ma = decay * loss_ma + ( 1 - decay ) * loss

        sys.stdout.write('\riter:%d/%d loss: %.5f'%(i,iters,loss_ma))
        sys.stdout.flush()

        if i % save_interval == 0 and save_loss:
            loss_history.append(loss)

  
    m.save(sess)

if save_loss:
    loss_file = 'data/loss'
    df = pd.DataFrame(loss_history,columns=['loss'])
    if os.path.isfile(loss_file):
        df_old = pd.read_pickle(loss_file)
        df = pd.concat([df_old,df],ignore_index=True)
    df.to_pickle(loss_file)

sys.stdout.write('\n')
sys.stdout.flush()
