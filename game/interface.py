from abc import ABC, abstractmethod
from copy import deepcopy


class Game(ABC):
    """
    Abstraction which presents the ruleset of a 2-player game with deterministic actions as states, actions and rewards.
    States are encoded as 1d numpy arrays of integers, and actions are encoded as integer indexes into the complete
    list of all actions.
    """

    def __init__(self, starting_player=0):
        self.starting_player = starting_player
        self.current_player = starting_player

    def create_copy(self):
        return deepcopy(self)

    @abstractmethod
    def get_initial_state(self):
        """
        Initial state of a game
        """
        pass

    @abstractmethod
    def get_current_state(self):
        pass

    @abstractmethod
    def get_child_state(self, action):
        """
        Resulting state when taking 'action' in 'state'
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
        Reward of the given state relative to the starting player (player 0)
        """
        pass

    def player_to_move(self):
        """
        Returns for a state which player should make a move next
        """
        return self.current_player

    @abstractmethod
    def next_player_to_move(self):
        """
        Returns for a state which player should make a move after a move is taken
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

    def visualize(self):
        """
        Visualize state
        """
        pass
