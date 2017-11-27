from game import GoGame
import agent
import sys

if len(sys.argv)>1:
    cycle = int(sys.argv[1])
else:
    cycle = 1e9

boardsize = 9

game = GoGame(boardsize)

agent = agent.Agent(eps=1.0/(1+cycle))

interactive = False

games_to_play = 1000

while True:

    if interactive:
        game.print_board()
        vertex = raw_input('%s plays:'%game.current_color)
        if vertex == 'agent':
            vertex = agent.play(game.current_color,game.legal_states())
    else:
        vertex = agent.play(game.current_color,game.legal_states())
    game.play(vertex) 

    if game.end:
        if interactive:
            play_more = raw_input('Play more?')
            if play_more == 'y':
                game.reset()
            else:
                break
        else:
            games_to_play -= 1
            game.save('./data/ep%d'%games_to_play)
            print '\rGames remaining:%d'%games_to_play,
            sys.stdout.flush()
            if games_to_play == 0:
                break
            else:
                game.reset()
            
        
