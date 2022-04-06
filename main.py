import copy
import os
from tqdm import tqdm
from yaml import safe_load, YAMLError
from game.hex import Hex
from game.nim import Nim
from rl.anet_agent import ANETAgent
from rl.topp import play_game, anet_tournament
from rl.uniform_agent import UniformAgent

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def sample_game(game, agent, num_games=1000, plot=False):
    wins = 0
    losses = 0
    progress = tqdm(range(num_games), desc="Game")
    for _ in progress:
        winner = play_game(game, agent, UniformAgent(), visualize=plot)
        if winner == 1:
            wins += 1
        elif winner == 2:
            losses += 1
        progress.set_description(
            "Wins: {} | Losses: {}".format(wins, losses) +
            " | WL-Ratio: {:.4f}% ".format((wins / (wins + losses)) * 100)
        )


def main():
    # load config
    config = dict()
    with open("config.yaml", "r") as stream:
        try:
            config = safe_load(stream)
        except YAMLError as exc:
            print(exc)

    # configure game
    # NOTE: no default values. expect user to make conscious decisions..
    game_config = config["game"]
    if game_config["name"] == "hex":
        game = Hex(**game_config["params"])
    elif game_config["name"] == "nim":
        game = Nim(**game_config["params"])
    else:
        raise ValueError(f'unknown game {game_config["name"]}')

    # configure agent
    agent_config = config.get("agent", {})
    agent = ANETAgent(config=agent_config, game=game)

    # train agent
    sample_game(game=game, agent=agent, plot=False)
    weight_files = agent.train()
    sample_game(game=game, agent=agent, plot=False)

    topp_config = config.get("topp", {})

    if topp_config.get("enabled", False):
        topp_num_sample_games = topp_config["num_games_per_series"]
        topp_include_uniform = topp_config["include_uniform"]
        topp_num_games_to_visualize_per_series = topp_config["num_games_to_visualize_per_series"]
        configs = []
        for weight_file in weight_files:
            cpy = copy.deepcopy(agent_config)
            cpy['anet']['weight_file'] = weight_file
            configs.append(cpy)
        anet_tournament(game, configs, topp_num_sample_games, topp_include_uniform, topp_num_games_to_visualize_per_series)


if __name__ == '__main__':
    main()
