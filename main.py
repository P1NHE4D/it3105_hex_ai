import os

import numpy as np
from yaml import safe_load, YAMLError

from game.hex import Hex
from game.nim import Nim
from rl.agent import Agent

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def sample_game(game, agent, num_games=100):
    wins = 0
    losses = 0
    for i in range(num_games):
        print(f"SAMPLE GAME {i}/{num_games}")
        state = game.get_initial_state()
        while not game.is_state_terminal(state):
            action = agent.propose_action(state, game.get_legal_actions(state))
            state = game.get_child_state(state, action)
            if game.is_state_terminal(state):
                break
            action_idx = np.random.choice(np.arange(len(game.get_legal_actions(state))))
            action = game.get_legal_actions(state)[action_idx]
            state = game.get_child_state(state, action)

        # viz final state
        game.visualize(state)

        reward = game.get_state_reward(state)
        if reward == 1.0:
            print("player 0 win!")
            wins += 1
        elif reward == -1.0:
            print("player 1 win!")
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
