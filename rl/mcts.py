class MCTSNode:
    def __init__(self):
        self.n = 0
        self.sum = 0


class MCTS:

    def __init__(self):
        pass

    def simulate(self, game, num_sim):
        """
        Simulates

        :param game: current game
        :param num_sim: number of simulations
        :return: distribution over all actions
        """
        # simulate from current state of the game and return distribution over all possible actions
        pass

    def retain_subtree(self, action):
        """

        :param action: action chosen from root
        """
        # retain chosen state but discard rest of tree
        pass
