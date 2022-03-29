from collections import defaultdict

from game.hex import Hex
from game.interface import Game
from rl.agent import Agent
from rl.anet_agent import ANETAgent
from rl.uniform_agent import UniformAgent


def play_game(game: Game, player_1: Agent, player_2: Agent, visualize=False):
    """
    Play game until completion, returning 1 if player 1 won, and 2 if player 2 won, and optionally visualizing the final
    game state
    """
    state = game.get_initial_state()
    player_to_move = player_1
    while True:
        action = player_to_move.propose_action(state, game.get_legal_actions(state))
        state = game.get_child_state(state, action)
        if game.is_state_terminal(state):
            if visualize:
                game.visualize(state)
            reward = game.get_state_reward(state)
            if reward == 0:
                raise ValueError('Game ended in tie, but ties are not implemented for play_game')
            return 1 if reward > 0 else 2
        player_to_move = player_2 if player_to_move == player_1 else player_1


def round_robin_tournament(game: Game, agents: list[tuple[str, ANETAgent]], num_games_per_series: int):
    """
    Each agent competes with each other agent playing num_games_per_series games. Then the wins/losses statistics of
    each player is returned
    """
    # wins & losses per agent, keyed on agent names
    wins = defaultdict(lambda: 0)
    losses = defaultdict(lambda: 0)

    for i, (a_name, a_agent) in enumerate(agents[:-1]):
        for b_name, b_agent in agents[i:]:
            for j in range(num_games_per_series):
                # in an attempt to be fair, alternate being the first player
                if j % 2 == 0:
                    winner_name = a_name if play_game(game, a_agent, b_agent) == 1 else b_name
                else:
                    winner_name = b_name if play_game(game, b_agent, a_agent) == 1 else a_name
                loser_name = a_name if winner_name == b_name else b_name

                wins[winner_name] += 1
                losses[loser_name] += 1

    return [
        (name, wins[name], losses[name])
        for name, _ in agents
    ]


if __name__ == '__main__':
    weight_files = [
        'rl/models/anet_episode_0',
        'rl/models/anet_episode_10',
        'rl/models/anet_episode_20',
        'rl/models/anet_episode_30',
        'rl/models/anet_episode_40',
        'rl/models/anet_episode_49',
    ]

    h = Hex(4)

    agents = [
        (weight_file, ANETAgent(game=h, config={'anet': {'weight_file': weight_file}}))
        for weight_file in weight_files
    ] + [
        ('uniform', UniformAgent()),
    ]

    play_game(h, agents[-2][1], agents[-1][1], visualize=True)

    results = round_robin_tournament(h, agents, 25)
    for name, wins, losses in results:
        print(name, wins/(wins+losses), wins, losses)