import numpy as np

from rl.agent import Agent


class UniformAgent(Agent):
    """
    Agent which picks random actions based on a uniform distribution
    """

    def propose_action(self, state, legal_actions):
        return np.random.choice(legal_actions)
