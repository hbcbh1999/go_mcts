from game import GoGame
import agent

boardsize = 9

game = GoGame(boardsize)

agent = agent.Agent()

interactive = False

games_to_play = 100

while True:

    if interactive:
        vertex = raw_input('%s plays:'%game.current_color)
        if vertex == 'agent':
            vertex = agent.play(game.current_color,game.legal_states)
    else:
        vertex = agent.play(game.current_color,game.legal_states)
    game.play(vertex) 

    if game.end:
        if interactive:
            play_more = raw_input('Play more?')
            if play_more == 'y':
                game.reset()
            else:
                break
        else:
            game.save()
            games_to_play -= 1
            game.reset()

