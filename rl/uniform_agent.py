import numpy as np

from rl.agent import Agent


class UniformAgent(Agent):
    """
    Agent which chooses proposes actions at random
    """

    def propose_action(self, state, legal_actions):
        return np.random.choice(legal_actions)
