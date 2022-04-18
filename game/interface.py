from abc import ABC, abstractmethod
from copy import deepcopy


class Game(ABC):
    """
    Abstraction which encapsulates a running (stateful) 2-player game with deterministic actions as states, actions and
    rewards. States are encoded as 1d numpy arrays of integers, and actions are encoded as integer indexes into the
    complete list of all actions.
    """

    def __init__(self, starting_player=0):
        self.starting_player = starting_player
        self.current_player = starting_player

    def create_copy(self):
        """
        Returns a deepcopy of the running game
        """
        return deepcopy(self)

    @abstractmethod
    def get_initial_state(self):
        """
        Resets the game to it's initial state, and returns this state
        """
        pass

    @abstractmethod
    def get_current_state(self):
        """
        Returns the current state of the game
        """
        pass

    @abstractmethod
    def get_child_state(self, action):
        """
        Take 'action' in the current state, returning the new state
        """
        pass

    @abstractmethod
    def is_state_terminal(self):
        """
        Returns True if the current state is terminal, and False otherwise. A state is terminal if the game is over
        """
        pass

    @abstractmethod
    def get_legal_actions(self):
        """
        Returns a list of legal actions
        """
        pass

    @abstractmethod
    def get_state_reward(self):
        """
        Reward of the current state relative to the starting player (player 0)
        """
        pass

    def player_to_move(self):
        """
        Returns which player should make a move next in the current state
        """
        return self.current_player

    @abstractmethod
    def next_player_to_move(self):
        """
        Returns which player should make a move after a move has been taken in the current state
        """
        pass

    @abstractmethod
    def number_of_actions(self):
        """
        Returns the length of the complete list of all actions. Given a response N, the caller can will know that
        actions are in the interval [0,N)
        """
        pass

    @abstractmethod
    def get_action(self, index):
        """
        :param index: index of action
        :return: action assigned to given index
        """
        pass

    def visualize(self, title=None, state=None):
        """
        Visualize current state
        """
        pass
