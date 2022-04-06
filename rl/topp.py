from collections import defaultdict

from tqdm import tqdm

from game.hex import Hex
from game.interface import Game
from rl.agent import Agent
from rl.anet_agent import ANETAgent
from rl.uniform_agent import UniformAgent


def play_game(game: Game, player_1: Agent, player_2: Agent, visualize=False, title=None):
    """
    Play game until completion, returning 1 if player 1 won, and 2 if player 2 won, and optionally visualizing the final
    game state (with an optional title)
    """
    state = game.get_initial_state()
    player_to_move = player_1
    while not game.is_state_terminal():
        action = player_to_move.propose_action(state, game.get_legal_actions())
        state = game.get_child_state(action)
        player_to_move = player_2 if player_to_move == player_1 else player_1

    reward = game.get_state_reward()
    if reward == 0:
        raise ValueError('Game ended in tie, but ties are not implemented for play_game')

    if visualize:
        game.visualize(title=title)

    return 1 if reward > 0 else 2


def round_robin_tournament(
        game: Game,
        agents: list[tuple[str, Agent]],
        num_games_per_series: int,
        num_games_to_visualize: int = 0
):
    """
    Each agent competes with each other agent playing num_games_per_series games. Then the wins/losses statistics of
    each player is returned
    """
    # wins & losses per agent, keyed on agent names
    wins = defaultdict(lambda: 0)
    losses = defaultdict(lambda: 0)

    num_series = len(agents) * (len(agents) - 1) / 2
    total_num_games = num_series * num_games_per_series

    with tqdm(total=total_num_games) as pbar:
        pbar.set_description("Playing tournament games")
        series = 0
        for i, (a_name, a_agent) in enumerate(agents[:-1]):
            series += 1
            for b_name, b_agent in agents[i + 1:]:
                for j in range(num_games_per_series):
                    pbar.update(1)
                    should_visualize = j < num_games_to_visualize
                    # in an attempt to be fair, alternate being the first player
                    if j % 2 == 0:
                        title = (f"SERIES {series} GAME {j}\n"
                                 f"PLAYER 1 (red): {a_name}\n"
                                 f"PLAYER 2 (black): {b_name}")
                        winner_name = a_name if play_game(game, a_agent, b_agent, should_visualize,
                                                          title) == 1 else b_name
                    else:
                        title = (f"SERIES {series} GAME {j}\n"
                                 f"PLAYER 1 (red): {b_name}\n"
                                 f"PLAYER 2 (black): {a_name}")
                        winner_name = b_name if play_game(game, b_agent, a_agent, should_visualize,
                                                          title) == 1 else a_name
                    loser_name = a_name if winner_name == b_name else b_name

                    wins[winner_name] += 1
                    losses[loser_name] += 1

    return [
        (name, wins[name], losses[name])
        for name, _ in agents
    ]


def anet_tournament(
        game: Game,
        anet_configs: list[dict],
        num_games_per_series: int,
        include_uniform: bool,
        num_games_to_visualize_per_series: int = 0,
):
    """
    Hold a game tournament of ANETAgents with weights loaded from provided weight files, optionally including a uniform
    agent as a sanity check
    """
    named_agents = [
        (anet_config['anet']['weight_file'], ANETAgent(game=game, config=anet_config))
        for anet_config in anet_configs
    ]
    if include_uniform:
        named_agents.append(('uniform', UniformAgent()))

    results = round_robin_tournament(game, named_agents, num_games_per_series, num_games_to_visualize_per_series)
    for name, wins, losses in results:
        print(f"{name} WL-Ratio {wins / (wins + losses)}, WINS {wins} LOSSES {losses}")


if __name__ == '__main__':
    weight_files = [
        'rl/models/anet_episode_0',
        'rl/models/anet_episode_10',
        'rl/models/anet_episode_20',
        'rl/models/anet_episode_30',
        'rl/models/anet_episode_40',
        'rl/models/anet_episode_49',
    ]

    configs = [
        {
            'anet': {
                'weight_file': weight_file,
                'hidden_layers': [[32, 'relu']],
            },
        }
        for weight_file in weight_files
    ]

    anet_tournament(
        game=Hex(3),
        anet_configs=configs,
        num_games_per_series=25,
        include_uniform=True,
    )
