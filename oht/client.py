from tqdm import tqdm

from game.hex import Hex
from oht.ActorClient import ActorClient
from rl.agent import Agent
import numpy as np


class OHTClient(ActorClient):

    def __init__(self, config, visualize=False):
        oht_config = config.get("oht", {})
        try:
            token = oht_config["token"]
        except KeyError:
            raise Exception("API token not found!")
        qualify = oht_config.get("qualify", False)
        super(OHTClient, self).__init__(auth=token, qualify=qualify)

        self.agent = None
        self.game = None
        self.board_size = None
        self.agent_config = config.get("agent", {})
        self.visualize = visualize
        self.progress: tqdm = None

    def handle_series_start(
            self, unique_id, series_id, player_map, num_games, game_params
    ):
        self.board_size = game_params[0]
        self.game = Hex(self.board_size)
        self.agent = Agent(config=self.agent_config, game=self.game)
        self.progress = tqdm(range(num_games), desc="Game")

        print("Series started | Number of games: {} | Board size: {}".format(num_games, self.board_size))
        print()

    def handle_get_action(self, state):
        state = self.encode_state(state)
        actions = self.game.get_legal_actions(state)
        action_idx = self.agent.propose_action(state, actions)
        row, col = self.game.get_action(action_idx)
        return row, col

    def handle_game_start(self, start_player):
        self.progress.set_description("Starting player: {}".format(start_player))
        self.progress.update(1)

    def handle_game_over(self, winner, end_state):
        if self.visualize:
            self.game.visualize(end_state)

    def handle_series_over(self, stats):
        for stat in stats:
            print("Player ID: {} | Player: {} | Wins: {} | Losses {}".format(*stat))

    def handle_tournament_over(self, score):
        print("Tournament score: {}".format(score))
        self.progress.close()

    def encode_state(self, state):
        encoded_state = np.zeros(2 + 2 * self.board_size ** 2)
        for i, cell in enumerate(state):
            if cell != 0:
                player_offset = 0 if cell == 1 else 1
                index = 2 * i + player_offset
                encoded_state[index] = 1
        return encoded_state
