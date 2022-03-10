from abc import ABC, abstractmethod
from copy import deepcopy


class Game(ABC):

    def create_copy(self):
        return deepcopy(self)

    @abstractmethod
    def init_game(self):
        """

        :return: state_encoding, actions
        """
        pass

    @abstractmethod
    def is_current_state_terminal(self):
        pass

    @abstractmethod
    def get_state_reward(self):
        # blablabla 1 if player 1 win, -1 if other guy wins, 0 otherwise (draw)
        pass

    @abstractmethod
    def get_child_state(self):
        """
        :return: state_encoding, actions
        """
        pass

    @abstractmethod
    def visualize(self):
        pass
