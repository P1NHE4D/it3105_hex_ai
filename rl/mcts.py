from pprint import pprint
import numpy as np
from game.interface import Game
from game.nim import Nim


class MCTSNode:
    def __init__(self, parent, action, player):
        """
        A MCTSNode represents a single node in the tree of MCTS, as well as the edge that leads to it (if any). The
        semantics are: "We were in 'parent_state', then 'edge_action' was taken, now I am in 'state' as 'player' and can
        choose from 'actions'"

        :param parent: state we were in to reach this state
        :param action: action that was taken from parent state to reach this state
        :param player: player to move
        """
        self.player = player
        self.parent = parent
        self.action = action
        self.node_visit_count = 0
        self.incoming_edge_visit_count = 0
        self.cumulative_reward = 0
        self.children: list[MCTSNode] = []

    def is_leaf(self):
        return len(self.children) == 0

    def describe(self):
        pprint(vars(self))

    def expand(self, game: Game):
        """
        Adds a new child node that has not been visited yet

        :param game: game instance
        :return: child node
        """
        for action in game.get_legal_actions():
            self.children.append(MCTSNode(
                parent=self,
                action=action,
                player=game.next_player_to_move()
            ))
        idx = np.random.choice(np.arange(len(self.children)))
        return self.children[idx]


class MCTS:

    def __init__(self, agent):
        self.root = None
        self.agent = agent

    def simulate(self, game: Game, num_sim):
        """
        Performs simulations of the current game to obtain an action distribution

        :param game: game instance
        :param num_sim: number of simulations
        :return: distribution over all actions
        """
        if self.root is None:
            self.root = MCTSNode(
                parent=None,
                action=None,
                player=game.player_to_move(),
            )
        for _ in range(num_sim):
            sim_game = game.create_copy()
            node: MCTSNode = self.root

            # use tree policy to get to a node that is not fully expanded and not a terminal node
            while not node.is_leaf():
                node = self.tree_policy(node)
                sim_game.get_child_state(node.action)

            # only expand if node is not terminal
            if not sim_game.is_current_state_terminal():
                node = node.expand(sim_game)
                sim_game.get_child_state(node.action)

            # obtain reward
            reward = self.rollout(sim_game)

            # update nodes on path
            while node is not None:
                node.node_visit_count += 1
                node.incoming_edge_visit_count += 1
                node.cumulative_reward += reward
                node = node.parent

        return self.action_distribution(game, self.root)

    def tree_policy(self, node, c=1.0):
        """
        Computes the optimal action in the given node and returns the resulting child node

        :param node: starting node
        :param c: exploration constant
        :return: optimal child node according to the tree policy
        """
        exploration_bonuses = []
        q_values = []
        for child in node.children:
            u = c * np.sqrt((np.log(node.node_visit_count) / (1 + child.incoming_edge_visit_count)))
            q = child.cumulative_reward / child.incoming_edge_visit_count if child.incoming_edge_visit_count > 0 else 0
            exploration_bonuses.append(u)
            q_values.append(q)
        child_idx = None

        if node.player == 0:
            # maximize
            child_idx = np.argmax(np.array(q_values) + np.array(exploration_bonuses))
        elif node.player == 1:
            # minimize
            child_idx = np.argmin(np.array(q_values) - np.array(exploration_bonuses))

        return node.children[child_idx]

    def rollout(self, game: Game):
        """
        Performs a rollout of the current game until a terminal state is reached

        :param game: game instance
        :return: obtained reward
        """
        state = game.get_current_state()
        while not game.is_current_state_terminal():
            action = self.agent.propose_action(state, game.get_legal_actions())
            state = game.get_child_state(action)
        return game.get_state_reward()

    def action_distribution(self, game: Game, node: MCTSNode):
        """
        Computes an action distribution over all actions for the given node
        :param game: game instance
        :param node: node for which the distribution should be calculated
        :return: distribution over all actions
        """
        distribution = np.zeros((game.get_action_length()))
        for child in node.children:
            idx = game.get_action_index(child.action)
            distribution[idx] = child.incoming_edge_visit_count
        return distribution / np.sum(distribution)

    def retain_subtree(self, action):
        """
        Picks the child node resulting from the given action as the new root of the tree

        :param action: action picked in root node
        """
        for child in self.root.children:
            if child.action == action:
                self.root = child
                self.root.parent = None
                return
        raise ValueError('passed action does not correspond to a child of root')


if __name__ == '__main__':
    # test case
    if False:
        game = Nim()
        node = MCTSNode(
            parent_state=None,
            edge_action=None,
            state=game.init_game(),
        )
        print(node.rollout(game))
    if False:
        game = Nim()
        node = MCTSNode(
            parent_state=None,
            edge_action=None,
            state=game.init_game(),
            player=game.player_to_move(),
            actions=game.get_legal_actions(),
        )
        while not node.is_leaf():
            node.describe()
            if not node.is_fully_expanded():
                node = node.expand(game)
        node.describe()
    if True:
        while True:
            game = Nim(
                stones=2,
                max_take=3,
            )
            game.init_game()
            mcts = MCTS()
            print(mcts.simulate(game, 10000))
