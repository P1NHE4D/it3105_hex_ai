from pprint import pprint
import numpy as np
from game.interface import Game
from game.nim import Nim


class MCTSNode:
    def __init__(self, parent, action, game_state, player, legal_actions, action_length):
        """
        A MCTSNode represents a single node in the tree of MCTS, as well as the edge that leads to it (if any). The
        semantics are: "We were in 'parent_state', then 'edge_action' was taken, now I am in 'state' as 'player' and can
        choose from 'actions'"

        :param parent: state we were in to reach this state
        :param action: action that was taken from parent state to reach this state
        :param game_state: state this node corresponds to
        :param player: player to move
        :param legal_actions: actions possible from node. Notice that len(actions) == 0 means the node is terminal
        :param action_length: length of single action representation. needed when building action distribution
        """
        self.game_state = game_state
        self.player = player
        self.parent = parent
        self.action = action
        self.action_length = action_length
        self.node_visit_count = 0
        self.incoming_edge_visit_count = 0
        self.cumulative_reward = 0
        self.children: list[MCTSNode] = []
        self.untried_actions = list(legal_actions)

    def is_terminal(self):
        return len(self.untried_actions) == 0 and len(self.children) == 0

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def describe(self):
        pprint(vars(self))

    def expand(self, game: Game):
        action = self.untried_actions.pop(0)
        child_state = game.get_child_state(action)
        child_node = MCTSNode(
            parent=self,
            action=action,
            game_state=child_state,
            player=game.current_player,
            legal_actions=game.get_legal_actions(),
            action_length=game.get_action_length(),
        )
        self.children.append(child_node)
        return child_node


class MCTS:

    def __init__(self, agent):
        self.root = None
        self.agent = agent

    def simulate(self, game: Game, num_sim):
        """
        Simulates

        :param game: current game
        :param num_sim: number of simulations
        :return: distribution over all actions
        """
        if self.root is None:
            self.root = MCTSNode(
                parent=None,
                action=None,
                game_state=game.get_current_state(),
                player=game.player_to_move(),
                legal_actions=game.get_legal_actions(),
                action_length=game.get_action_length(),
            )
        for _ in range(num_sim):
            sim_game = game.create_copy()
            node: MCTSNode = self.root

            # use tree policy to get to a node that is not fully expanded
            while node.is_fully_expanded() and not node.is_terminal():
                node = self.tree_policy(node)
                sim_game.get_child_state(node.action)

            # only expand if node is not terminal
            if not node.is_terminal():
                node = node.expand(sim_game)

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
        exploration_bonuses = []
        q_values = []
        for child in node.children:
            u = c * np.sqrt((np.log(node.node_visit_count) / (1 + child.incoming_edge_visit_count)))
            q = child.cumulative_reward / child.incoming_edge_visit_count
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
        state = game.get_current_state()
        while not game.is_current_state_terminal():
            action = self.agent.propose_action(state, game.get_legal_actions())
            state = game.get_child_state(action)
        return game.get_state_reward()

    def action_distribution(self, game: Game, node: MCTSNode):
        distribution = np.zeros((game.get_action_length()))
        for child in node.children:
            idx = game.get_action_index(child.action)
            distribution[idx] = child.incoming_edge_visit_count
        return distribution / np.sum(distribution)

    def retain_subtree(self, action):
        """
        Retain chosen state but discard rest of tree

        :param action: action chosen from root
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
