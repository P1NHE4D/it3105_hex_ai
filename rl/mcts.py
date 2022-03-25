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


class MCTS:

    def __init__(self, config, agent, game):
        self.root = None
        self.agent = agent
        self.c = config.get("c", 1.0)
        self.exp_prob = config.get("exp_prob", 1.0)
        self.game = game

    def simulate(self, state, num_sim):
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
                player=self.game.player_to_move(state),
            )
        for _ in range(num_sim):
            simulation_state = state

            node: MCTSNode = self.root

            # use tree policy to get to a node that is not fully expanded and not a terminal node
            while not node.is_leaf():
                node = self.tree_policy(node)
                simulation_state = self.game.get_child_state(simulation_state, node.action)

            # only expand if node is not terminal
            expand_tree = np.random.choice([True, False], p=[self.exp_prob, 1 - self.exp_prob])
            if expand_tree and not self.game.is_state_terminal(simulation_state):
                node = self.expand(node, simulation_state)
                simulation_state = self.game.get_child_state(simulation_state, node.action)

            # obtain reward
            reward = self.rollout(simulation_state)

            # update nodes on path
            while node is not None:
                node.node_visit_count += 1
                node.incoming_edge_visit_count += 1
                node.cumulative_reward += reward
                node = node.parent

        return self.action_distribution(state, self.root)

    def tree_policy(self, node):
        """
        Computes the optimal action in the given node and returns the resulting child node

        :param node: starting node
        :param c: exploration constant
        :return: optimal child node according to the tree policy
        """
        exploration_bonuses = []
        q_values = []
        for child in node.children:
            u = self.c * np.sqrt((np.log(node.node_visit_count) / (1 + child.incoming_edge_visit_count)))
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

    def rollout(self, state):
        """
        Performs a rollout of the current game until a terminal state is reached

        :param game: game instance
        :return: obtained reward
        """
        while not self.game.is_state_terminal(state):
            action = self.agent.propose_action(state, self.game.get_legal_actions(state))
            state = self.game.get_child_state(state, action)
        return self.game.get_state_reward(state)

    def action_distribution(self, state, node: MCTSNode):
        """
        Computes an action distribution over all actions for the given node
        :param state: game state
        :param node: node for which the distribution should be calculated
        :return: distribution over all actions
        """
        distribution = np.zeros((self.game.number_of_actions()))
        for child in node.children:
            idx = child.action
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

    def expand(self, node, state):
        """
        Expands node

        :param node: node to expand
        :param state: game state
        :return: child node
        """
        for action in self.game.get_legal_actions(state):
            node.children.append(MCTSNode(
                parent=node,
                action=action,
                player=self.game.next_player_to_move(state)
            ))
        idx = np.random.choice(np.arange(len(node.children)))
        return node.children[idx]




if __name__ == '__main__':
    # test case
    if True:
        game = Nim(
            stones=4,
            max_take=2,
        )
        mcts = MCTS(
            config={},
            agent=None,
            game=game,
        )
        print(mcts.simulate(game.get_initial_state(), num_sim=5000))
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
    if False:
        while True:
            game = Nim(
                stones=2,
                max_take=3,
            )
            game.init_game()
            mcts = MCTS()
            print(mcts.simulate(game, 10000))
