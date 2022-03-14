from abc import ABC, abstractmethod
from copy import deepcopy


class Game(ABC):

    def __init__(self, current_player=0, max_players=2):
        self.current_player = current_player
        self.max_players = max_players

    def create_copy(self):
        return deepcopy(self)

    @abstractmethod
    def init_game(self):
        """

        :return: state_encoding
        """
        pass

    @abstractmethod
    def is_current_state_terminal(self):
        pass

    @abstractmethod
    def get_actions(self):
        """
        :return: possible actions
        """
        pass

    @abstractmethod
    def get_child_state(self, action):
        """
        :return: state_encoding
        """
        pass

    @abstractmethod
    def get_state_reward(self):
        # blablabla 1 if player 1 win, -1 if other guy wins, 0 otherwise (draw)
        pass

    @abstractmethod
    def visualize(self):
        pass

    def player_to_move(self):
        """

        :return: which player's turn it is. 0 or 1
        """
        return self.current_player

    def next_player(self):
        self.current_player = (self.current_player + 1) % self.max_players
