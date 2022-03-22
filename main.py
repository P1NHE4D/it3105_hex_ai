import os
from yaml import safe_load, YAMLError

from game.hex import Hex
from game.nim import Nim
from rl.agent import Agent

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    config = dict()
    with open("config.yaml", "r") as stream:
        try:
            config = safe_load(stream)
        except YAMLError as exc:
            print(exc)
    nim_config = {'stones': 5, 'max_take': 3}
    game = Nim(**nim_config)
    agent = Agent(config=config.get("agent", {}), game=game)
    agent.train()

    wins = 0
    losses = 0
    for _ in range(10):
        game = Nim(**nim_config)
        state = game.init_game()
        while not game.is_current_state_terminal():
            action = agent.propose_action(state, game.get_legal_actions())
            print("Player: {} | Action: {} | Remaining stones: {}".format(game.player_to_move(), action, game.remaining_stones))
            game.get_child_state(action)
        reward = game.get_state_reward()
        if reward == 1.0:
            wins += 1
        elif reward == -1.0:
            losses += 1
    print("wins {} | losses {}".format(wins, losses))


if __name__ == '__main__':
    main()
