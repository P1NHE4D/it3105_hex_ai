import os

import numpy as np
from yaml import safe_load, YAMLError

from game.hex import Hex
from game.nim import Nim
from rl.agent import Agent

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def sample_game(game, agent):
    wins = 0
    losses = 0
    for _ in range(10):
        state = game.init_game()
        while not game.is_current_state_terminal():
            action = agent.propose_action(state, game.get_legal_actions())
            game.get_child_state(action)
            game.visualize()
            if game.is_current_state_terminal():
                break
            action_idx = np.random.choice(np.arange(len(game.get_legal_actions())))
            action = game.get_legal_actions()[action_idx]
            state = game.get_child_state(action)
            game.visualize()
        reward = game.get_state_reward()
        if reward == 1.0:
            wins += 1
        elif reward == -1.0:
            losses += 1
    print("wins {} | losses {}".format(wins, losses))


def main():
    config = dict()
    with open("config.yaml", "r") as stream:
        try:
            config = safe_load(stream)
        except YAMLError as exc:
            print(exc)
    game = Hex(3)
    agent = Agent(config=config.get("agent", {}), game=game)

    # sample_game(game, agent)

    agent.train()

    sample_game(game, agent)


if __name__ == '__main__':
    main()
