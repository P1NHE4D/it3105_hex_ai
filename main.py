import os
from yaml import safe_load, YAMLError

from game.hex import Hex
from rl.agent import Agent

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    config = dict()
    try:
        config: dict = safe_load("config.yaml")
    except YAMLError as exc:
        print(exc)
    game = Hex()
    agent = Agent(config=config, game=game)
    agent.train()


if __name__ == '__main__':
    main()
