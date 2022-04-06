from pprint import pprint
import numpy as np
from game.interface import Game
from rl.lite_model import LiteModel


class MCTSNode:
    def __init__(self, parent, action, player):
        """
        An MCTSNode represents a single node in the tree of MCTS, as well as the edge that leads to it (if any). The
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
        """
        :return: true if node is a leaf node, false otherwise
        """
        return len(self.children) == 0


def action_distribution(node: MCTSNode, num_actions):
    """
    Computes an action distribution over all actions for the given node

    :param node: node for which the distribution should be calculated
    :param num_actions: number of actions in the game
    :return: distribution over all actions
    """
    distribution = np.zeros(num_actions)
    for child in node.children:
        idx = child.action
        distribution[idx] = child.incoming_edge_visit_count
    return distribution / np.sum(distribution)


def expand(node: MCTSNode, game: Game):
    """
    Expands node

    :param node: node to expand
    :param game: game instance
    :return: child node
    """
    for action in game.get_legal_actions():
        node.children.append(MCTSNode(
            parent=node,
            action=action,
            player=game.next_player_to_move()
        ))
    idx = np.random.choice(np.arange(len(node.children)))
    return node.children[idx]


class MCTS:
    def __init__(
            self,
            config,
            default_policy,
            epsilon,
            critic,
            sigma
    ):
        """
        Constructs an MCTS tree that is used to simulate a number of games from a given state in order to
        pick the best action in that state

        :param config: mcts config
        :param default_policy: must be a lambda taking a state and a list of legal actions as input, and returning a single action from that list
        :param epsilon: probability for picking a random action during rollout
        :param critic: critic network
        :param sigma: probability for using rollouts instead of the critic
        """
        self.root = None
        self.c = config.get("c", 1.0)
        self.exp_prob = config.get("exp_prob", 1.0)
        self.default_policy = default_policy
        self.epsilon = epsilon
        self.critic: LiteModel = critic
        self.sigma = sigma

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
            expand_tree = np.random.random() < self.exp_prob
            if expand_tree and not sim_game.is_state_terminal():
                node = expand(node, sim_game)
                sim_game.get_child_state(node.action)

            # obtain reward
            if np.random.random() < self.sigma:
                reward = self.rollout(sim_game)
            else:
                state = sim_game.get_current_state()
                reward = self.critic.predict(np.array([state]))[0][0]

            # update nodes on path
            while node is not None:
                node.node_visit_count += 1
                node.incoming_edge_visit_count += 1
                node.cumulative_reward += reward
                node = node.parent

        return action_distribution(self.root, game.number_of_actions())

    def tree_policy(self, node):
        """
        Computes the optimal action in the given node and returns the resulting child node

        :param node: starting node
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

    def rollout(self, game):
        """
        Performs a rollout of the current game until a terminal state is reached.

        :param game: perform rollout from state of the given game
        :return: obtained reward
        """
        state = game.get_current_state()
        while not game.is_state_terminal():
            if np.random.random() < self.epsilon:
                action = np.random.choice(game.get_legal_actions())
            else:
                action = self.default_policy(state, game.get_legal_actions())
            state = game.get_child_state(action)
        return game.get_state_reward()

    def retain_subtree(self, action):
        """
        Picks the child node resulting from the given action as the new root of the tree

        :param action: action picked in root node
        """
        for child in self.root.children:
            if child.action == action:
                self.root = child
                self.root.parent = None
                self.root.action = None
                return
        raise ValueError('passed action does not correspond to a child of root')
