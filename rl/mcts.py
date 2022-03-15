"""
I admit I had a look at this source for inspiration: https://ai-boson.github.io/mcts/
"""
from pprint import pprint

import numpy as np

from game.interface import Game

from game.nim import Nim

class MCTSNode:
    def __init__(self, parent_state, edge_action, state, player, actions):
        """
        A MCTSNode represents a single node in the tree of MCTS, as well as the edge that leads to it (if any). The
        semantics are: "We were in 'parent_state', then 'edge_action' was taken, now I am in 'state' as 'player' and can
        choose from 'actions'"

        :param parent_state: state we were in to reach this state
        :param edge_action: action that was taken from parent state to reach this state
        :param state: state this node corresponds to
        :param player: player to move
        :param actions: actions possible from node. Notice that len(actions) == 0 means the node is terminal
        """
        self.state = state
        self.parent_state = parent_state
        self.incoming_edge_action = edge_action
        self.player = player

        # number of times node has been visited during tree search
        self.node_visit_count = 0
        # number of times the edge leading to this node has been visited during tree search
        self.incoming_edge_visit_count = 0
        # cumulative reward we've received following the edge leading to this node
        self.incoming_edge_cumulative_reward = 0
        # notice that Q can be calculated from the latter two properties

        # children reachable from node through actions
        self.children : list[MCTSNode] = []
        self.untried_actions = actions

    def rollout_policy(self, actions):
        # TODO I don't think we should be moving randomly here..
        return actions[np.random.randint(len(actions))]

    def rollout(self, game: Game):
        """
        Perform rollout simulation from this node, which means digging into state space until reaching a terminal
        node, and returning the reward of that node

        :param game: game object that MUST be at state self.state
        :return: reward at terminal state
        """
        while not game.is_current_state_terminal():
            actions = game.get_actions()
            chosen_action = self.rollout_policy(actions)
            _ = game.get_child_state(chosen_action)
        return game.get_state_reward()

    def backpropagate(self, reward):
        self.node_visit_count += 1
        self.incoming_edge_visit_count += 1
        self.incoming_edge_cumulative_reward += reward
        if self.parent_state is not None:
            self.parent_state.backpropagate(reward)

    def terminal_node(self):
        # if we've tried all actions and there are no children this is surely terminal
        return len(self.untried_actions) == 0 and len(self.children) == 0

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def describe(self):
        pprint(vars(self))

    def expand(self, game: Game):
        # it's critical that we expand in the order of left to right here, so that the children lie in the same order
        # as the 'actions' param given in __init__
        action = self.untried_actions.pop(0)
        child_state = game.get_child_state(action)
        child_node = MCTSNode(
            parent_state=self,
            edge_action=action,
            state=child_state,
            player=game.current_player,
            actions=game.get_actions(),
        )
        self.children.append(child_node)
        return child_node

    def action_distribution(self):
        edge_visit_counts = []
        for child in self.children:
            edge_visit_counts.append(child.incoming_edge_visit_count)

        # add 0s for untried actions as well (children we never got to see)
        for _ in self.untried_actions:
            edge_visit_counts.append(0)

        total = sum(edge_visit_counts)

        distribution = []
        for edge_visit_count in edge_visit_counts:
            distribution.append(edge_visit_count / total)

        return distribution

    def best_child(self, c=1.0):
        exploration_bonuses = []
        q_values = []
        for child in self.children:
            exploration_bonuses.append(c * (np.log(self.node_visit_count) / (1 + child.incoming_edge_visit_count)))
            q_values.append(child.incoming_edge_cumulative_reward / child.incoming_edge_visit_count)

        child_idx = None
        if self.player == 0:
            # maximize
            child_idx = np.argmax(np.array(q_values) + np.array(exploration_bonuses))
        elif self.player == 1:
            # minimize
            child_idx = np.argmin(np.array(q_values) - np.array(exploration_bonuses))

        return self.children[child_idx]

class MCTS:

    def __init__(self):
        self.root : MCTSNode = None

    def simulate(self, game: Game, num_sim):
        """
        Simulates

        :param game: current game
        :param num_sim: number of simulations
        :return: distribution over all actions
        """
        # this might be the first time the class is used. Initialize the tree from scratch in that case
        if self.root is None:
            # make new MCTS tree (root has no parent)
            self.root = MCTSNode(
                parent_state=None,
                edge_action=None,
                state=game.get_current_state(),
                player=game.player_to_move(),
                actions=game.get_actions(),
            )

        # simulate from current state of the game and return distribution over all possible actions
        for _ in range(num_sim):
            simulation_game = game.create_copy()
            leaf = self.tree_search(simulation_game)
            # ...simulation_game should have been mutated now to reflect state in leaf...
            reward = leaf.rollout(simulation_game)
            leaf.backpropagate(reward)

        return self.root.action_distribution()

    def tree_search(self, game: Game):
        node = self.root
        while not node.terminal_node():
            if not node.is_fully_expanded():
                return node.expand(game)
            else:
                node = node.best_child()
                game.get_child_state(node.incoming_edge_action)

        return node

    def retain_subtree(self, action):
        """

        :param action: action chosen from root
        """
        # retain chosen state but discard rest of tree
        pass

if __name__ == '__main__':
    # if ran directly, run test case

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
            actions=game.get_actions(),
        )
        while not node.terminal_node():
            node.describe()
            if not node.is_fully_expanded():
                node = node.expand(game)
        node.describe()
    if True:
        while True:
            game = Nim(
                stones=14,
                max_take=3,
            )
            game.init_game()
            mcts = MCTS()
            print(mcts.simulate(game, 10000))