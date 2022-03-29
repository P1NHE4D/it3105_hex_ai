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
        self.config = config
        self.visualize = visualize

    def handle_series_start(
            self, unique_id, series_id, player_map, num_games, game_params
    ):
        board_size = game_params[0]
        self.game = Hex(board_size)
        self.agent = Agent(config=self.config, game=self.game)

        print("Series started | Number of games: {} | Board size: {}".format(num_games, board_size))

    def handle_get_action(self, state):
        actions = self.game.get_legal_actions(state)
        distribution = self.agent.propose_action(state, actions)
        action_idx = np.argmax(distribution)
        row, col = self.game.get_action(action_idx)
        return row, col

    def handle_game_over(self, winner, end_state):
        if self.visualize:
            self.game.visualize(end_state)

    def handle_series_over(self, stats):
        print("Series stats: {}".format(stats))

    def handle_tournament_over(self, score):
        print("Tournament score: {}".format(score))
