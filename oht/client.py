from tqdm import tqdm

from game.hex import Hex
from oht.ActorClient import ActorClient
import numpy as np

from rl.anet_agent import ANETAgent


class OHTClient(ActorClient):

    def __init__(self, config):
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
        self.visualize = oht_config.get("visualize", False)
        self.progress: tqdm = None
        self.played_games = []

    def handle_series_start(
            self, unique_id, series_id, player_map, num_games, game_params
    ):
        self.board_size = game_params[0]
        self.game = Hex(self.board_size)
        self.agent = ANETAgent(config=self.agent_config, game=self.game)
        self.progress = tqdm(range(num_games), desc="Game")

        print("Series started | Number of games: {} | Board size: {}".format(num_games, self.board_size))
        print()

    def handle_get_action(self, state):
        actions = [i for i, cell in enumerate(state[1:]) if cell == 0]
        encoded_state = self.encode_state(state, transpose=True)
        action_idx = self.agent.propose_action(encoded_state, actions)
        row, col = self.game.get_action(action_idx)
        return row, col

    def handle_game_start(self, start_player):
        self.progress.set_description("Starting player: {}".format(start_player))
        self.progress.update(1)

    def handle_game_over(self, winner, end_state):
        if self.visualize:
            self.played_games.append(end_state)

    def handle_series_over(self, stats):
        for stat in stats:
            print("Player ID: {} | Player: {} | Wins: {} | Losses {}".format(*stat))

    def handle_tournament_over(self, score):
        print("Tournament score: {}".format(score))
        self.progress.close()
        if self.visualize:
            for state in self.played_games:
                self.game.visualize(state=self.encode_state(state))

    def encode_state(self, state, transpose=False):
        if transpose and state[0] == 2:
            state = np.array(list(map(lambda x: 1 if x == 2 else 2 if x == 1 else 0, state)))
        encoded_state = np.zeros(2 + 2 * self.board_size ** 2)
        for i, cell in enumerate(state):
            if cell != 0:
                player_offset = 0 if cell == 1 else 1
                index = 2 * i + player_offset
                encoded_state[index] = 1
        return encoded_state
