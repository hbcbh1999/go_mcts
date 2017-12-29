from game import GoGame
#import agent
from agent_mcts import Agent
import sys
import numpy as np
import argparse
import os.path
import random
import pandas as pd
"""
ARGUMENTS
"""
parser = argparse.ArgumentParser()
parser.add_argument('--interactive', default=False, help='Text interactive interface', action='store_true')
parser.add_argument('--cycle', default=-1, type=int, help='Number of cycle')
parser.add_argument('--black', default='random', choices=['agent','random'], help='AI for black')
parser.add_argument('--white', default='random', choices=['agent','random'], help='AI for white')
parser.add_argument('--ngames', default=500, type=int, help='Number of episodes to play')
parser.add_argument('--save', default=False, help='Save self-play episodes', action='store_true')
parser.add_argument('--save_dir', default='./data',type=str,help='Directory for save')
args = parser.parse_args()

interactive = args.interactive
cycle = args.cycle
black = args.black
white = args.white
ngames = args.ngames
save = args.save
save_dir = args.save_dir

"""
SOME INITS
"""
boardsize = 9

game = GoGame(boardsize)

if black == 'agent' or white == 'agent' or interactive:
    if cycle == -1:
        agent = Agent(0.5)
    else:
        agent = Agent(0.5)

if save:
    df = pd.DataFrame(columns=['board','color','policy','result'])
    df_ep = pd.DataFrame(columns=['board','color','policy'])
    
ai_color = {'black':black, 'white':white}

game_results = [0,0,0]

"""
MAIN GAME LOOP
"""
while True:

    print(game.env.board)
    if interactive:
        game.print_board()
        vertex = raw_input('%s plays:'%game.current_color)
        if vertex == 'agent':
            vertex = agent.play(game.current_color,game.legal_states())
    else:
        legal_states = game.legal_states()
        ai_type = ai_color[game.current_color]
        if ai_type == 'agent':
            vertex = agent.play(game)
            if save:
                _dict = {'board':game.env.board,'color':game.current_color,'policy':agent.get_prob()}
                df_ep = df_ep.append(_dict,ignore_index=True)
        elif ai_type == 'random':
            vertex = random.choice(list(legal_states.keys()))
            
    game.play(vertex) 

    if game.end:
        if interactive:
            play_more = raw_input('Play more?')
            if play_more == 'y':
                game.reset()
            else:
                break
        else:
            if game.winner == 'black':
                game_results[0] += 1
            elif game.winner == 'white':
                game_results[1] += 1
            else:
                game_results[2] += 1
            ngames -= 1
            if save:
                if game.winner == 'black':
                    r_map = {'black':1, 'white':0}
                elif game.winner == 'white':
                    r_map = {'black':0, 'white':1}
                else:
                    r_map = {'black':0.5, 'white':0.5}
                df_ep['result'] = df_ep['color'].map(r_map)
                df = df.append(df_ep)
            sys.stdout.write('\rGames remaining:%d Results(b/w/d):%d/%d/%d'%(ngames,game_results[0],game_results[1],game_results[2]))
            sys.stdout.flush()
            if ngames == 0:
                break
            else:
                game.reset()


df.to_pickle('test.pickle')
