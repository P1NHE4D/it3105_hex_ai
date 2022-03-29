import os

import numpy as np
from tqdm import tqdm
from yaml import safe_load, YAMLError

from game.hex import Hex
from game.nim import Nim
from rl.anet_agent import ANETAgent
from rl.topp import play_game
from rl.uniform_agent import UniformAgent

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def sample_game(game, agent, num_games=100, plot=False):
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
    config = dict()
    with open("config.yaml", "r") as stream:
        try:
            config = safe_load(stream)
        except YAMLError as exc:
            print(exc)
    game = Hex(7)
    agent = ANETAgent(config=config.get("agent", {}), game=game)

    sample_game(game=game, agent=agent, plot=False)

    agent.train()

    sample_game(game=game, agent=agent, plot=False)


if __name__ == '__main__':
    main()
