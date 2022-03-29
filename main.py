import os

import numpy as np
from tqdm import tqdm
from yaml import safe_load, YAMLError

from game.hex import Hex
from game.nim import Nim
from rl.agent import Agent

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def sample_game(game, agent, num_games=100, plot=False):
    wins = 0
    losses = 0
    progress = tqdm(range(num_games), desc="Game")
    for _ in progress:
        state = game.get_initial_state()
        while not game.is_state_terminal(state):
            action = agent.propose_action(state, game.get_legal_actions(state))
            state = game.get_child_state(state, action)
            if game.is_state_terminal(state):
                break
            action_idx = np.random.choice(np.arange(len(game.get_legal_actions(state))))
            action = game.get_legal_actions(state)[action_idx]
            state = game.get_child_state(state, action)

        if plot:
            game.visualize(state)

        reward = game.get_state_reward(state)
        if reward == 1.0:
            wins += 1
        elif reward == -1.0:
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
    game = Hex(6)
    agent = Agent(config=config.get("agent", {}), game=game)

    sample_game(game=game, agent=agent, plot=False)

    agent.train()

    sample_game(game=game, agent=agent, plot=False)


if __name__ == '__main__':
    main()
