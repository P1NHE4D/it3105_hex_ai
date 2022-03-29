from abc import ABC, abstractmethod


class Agent(ABC):
    """
    The abstract notion of an agent: something that can propose an action given a state and a set of legal actions
    """

    @abstractmethod
    def propose_action(self, state, legal_actions):
        pass
