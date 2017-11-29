from game import GoGame
import agent
import sys
import numpy as np
import argparse
"""
ARGUMENTS
"""
parser = argparse.ArgumentParser()
parser.add_argument('--interactive', help='Text interactive interface', action='store_true')
parser.add_argument('--cycle', type=int, help='Number of cycle')
parser.add_argument('--black', choices=['agent','random'], help='AI for black')
parser.add_argument('--white', choices=['agent','random'], help='AI for white')
parser.add_argument('--ngames', type=int, help='Number of episodes to play')
args = parser.parse_args()

interactive = args.interactive

if args.cycle:
    cycle = args.cycle

if args.black:
    black = args.black
else:
    black = 'random'

if args.white:
    white = args.white
else:
    white = 'random'

if args.ngames:
    ngames = args.ngames
else:
    ngames = 500


"""
SOME INITS
"""
boardsize = 9

game = GoGame(boardsize)

#agent = agent.Agent(eps=1.0/(1+cycle))
agent = agent.Agent(eps=0)

ai_color = {'black':black, 'white':white}

game_results = [0,0,0]

"""
MAIN GAME LOOP
"""
while True:

    if interactive:
        game.print_board()
        vertex = raw_input('%s plays:'%game.current_color)
        if vertex == 'agent':
            vertex = agent.play(game.current_color,game.legal_states())
    else:
        legal_states = game.legal_states()
        ai_type = ai_color[game.current_color]
        if ai_type == 'agent':
            vertex = agent.play(game.current_color,legal_states)
        elif ai_type == 'random':
            _tmp = np.random.randint(len(legal_states))
            vertex = legal_states[_tmp][0]
            
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
            game.save('./data/ep%d'%ngames)
            print '\rGames remaining:%d Results(b/w/d):%d/%d/%d'%(games_to_play,game_results[0],game_results[1],game_results[2]),
            sys.stdout.flush()
            if games_to_play == 0:
                break
            else:
                game.reset()
            
        
