from enum import Enum
from game.interface import Game
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class HexCellState(Enum):
    EMPTY = 0
    PLAYER_ONE = 1
    PLAYER_TWO = 2


class HexState(Enum):
    UNDECIDED = 0
    PLAYER_ONE_WON = 1
    PLAYER_TWO_WON = 2


class HexCell:

    def __init__(self, neighbours):
        self.neighbours = neighbours
        self.state = HexCellState.EMPTY


class Hex(Game):

    def __init__(self, board_size):
        super().__init__()
        self.board = construct_hex_board(board_size)
        self.ohe_board: list = []
        self.board_size = board_size
        self.legal_actions: list = []
        self.all_actions = [(row, col) for row in range(self.board_size) for col in range(self.board_size)]
        self.state = HexState.UNDECIDED

    def init_game(self):
        """
        Resets the game board and returns a one-hot encoded representation of the initial state

        One hot encoding representation:
        - first two digits indicate current player: [10...] player one, [01....] player two
        - remaining digits encode each cell state with 2 bits per cell: [...00...] empty,
          [...10...] player one, [...01...] player two
        - length of ohe vector is 2 + 2 * board_size * board_size

        :return: one-hot encoded representation of the initial state
        """
        # reset all cells of the board
        for node in self.board.flatten():
            node.game_state = HexCellState.EMPTY

        # initialize new one hot encoding
        self.ohe_board = np.zeros(2 + 2 * self.board_size ** 2)
        self.ohe_board[self.current_player] = 1

        # reset actions
        self.legal_actions = self.all_actions.copy()

        # reset game state
        self.state = HexState.UNDECIDED

        return self.ohe_board

    def get_current_state(self):
        return self.ohe_board

    def is_current_state_terminal(self):
        """
        :return: True if terminal, otherwise False
        """
        return self.state == HexState.PLAYER_ONE_WON or self.state == HexState.PLAYER_TWO_WON

    def get_legal_actions(self):
        """
        :return: returns array comprising possible actions in current state
        """
        return self.legal_actions

    def get_action_length(self):
        return len(self.all_actions)

    def get_action_index(self, action):
        row, col = action
        return row * self.board_size + col

    def get_action_by_index(self, index):
        return self.all_actions[index]

    def get_child_state(self, action):
        """
        Determines the child state based on the chosen action, returning
        a one-hot encoded successor state.

        :param action: picked action
        :return: one-hot encoded successor state
        """

        # set board state
        row, col = action
        self.board[row, col].game_state = HexCellState.PLAYER_ONE if self.current_player == 0 else HexCellState.PLAYER_TWO

        # remove picked action from possible actions
        self.legal_actions.remove((row, col))

        self.update_game_state(row, col)

        # update one hot encoding
        ohe_index = 2 + 2 * (row * self.board_size + col)
        if self.current_player == 1:
            ohe_index += 1
        self.ohe_board[ohe_index] = 1
        self.ohe_board[self.current_player] = 0
        self.advance_player()
        self.ohe_board[self.current_player] = 1

        return self.ohe_board

    def update_game_state(self, row, col):
        """
        Checks if any player achieved building a connection from one end to the other
        and updates the game state accordingly.
        Conducting a check after each turn is computationally more efficient than checking
        the entire board upon calling the is_state_terminal function.

        :param row: row of the placed piece
        :param col: column of the placed piece
        """

        def connected_lines(row, col, visited=None):
            """
            Computes the number of connected lines (rows if player 1, columns if player 2)

            :param row:
            :param col:
            :param visited: visited nodes
            :return: returns a set containing the indices of the connected lines
            """

            if visited is None:
                visited = set()

            lines = set()
            cell_val = HexCellState.PLAYER_ONE if self.current_player == 0 else HexCellState.PLAYER_TWO
            if cell_val == HexCellState.PLAYER_ONE:
                lines.add(row)
            else:
                lines.add(col)
            visited.add((row, col))
            for neighbour in self.board[row, col].neighbours:
                neighbour_row, neighbour_col = neighbour
                if self.board[neighbour_row, neighbour_col].game_state == cell_val and (
                neighbour_row, neighbour_col) not in visited:
                    lines = lines.union(connected_lines(neighbour_row, neighbour_col, visited))

            return lines

        # player one wins if pieces are connected over all rows
        # player two wins if pieces are connected over all columns
        if len(connected_lines(row, col)) == self.board_size:
            if self.current_player == 0:
                self.state = HexState.PLAYER_ONE_WON
            else:
                self.state = HexState.PLAYER_TWO_WON

    def get_state_reward(self):
        if self.state == HexState.PLAYER_ONE_WON:
            return 1
        if self.state == HexState.PLAYER_TWO_WON:
            return -1
        return 0

    def visualize(self):
        g = nx.Graph()
        for row, hex_row in enumerate(self.board):
            for col, node in enumerate(hex_row):
                node_color = "white"
                node_edge_color = "black"
                if node.game_state == HexCellState.PLAYER_ONE:
                    node_color = "red"
                    node_edge_color = "red"
                elif node.game_state == HexCellState.PLAYER_TWO:
                    node_color = "black"
                g.add_node("node_{}_{}".format(row, col), color=node_color, edge_color=node_edge_color)

        # cannot be integrated into the previous loop as this will mix up the order of the nodes
        for row, hex_row in enumerate(self.board):
            for col, node in enumerate(hex_row):
                for neighbour in node.neighbours:
                    edge_color = "black"
                    edge_weight = 1
                    if node.game_state == self.board[neighbour].game_state:
                        if node.game_state == HexCellState.PLAYER_ONE:
                            edge_color = "red"
                            edge_weight = 4
                        elif node.game_state == HexCellState.PLAYER_TWO:
                            edge_weight = 4
                    g.add_edge("node_{}_{}".format(row, col), "node_{}_{}".format(*neighbour), weight=edge_weight,
                               color=edge_color)

        node_colors = nx.get_node_attributes(g, 'color').values()
        node_edge_colors = nx.get_node_attributes(g, 'edge_color').values()
        edge_colors = nx.get_edge_attributes(g, 'color').values()
        edge_weights = nx.get_edge_attributes(g, 'weight').values()
        nx.draw(
            g,
            edge_color=edge_colors,
            width=list(edge_weights),
            edgecolors=list(node_edge_colors),
            node_color=node_colors,
            pos=diamond_layout(g)
        )
        plt.show()


def diamond_layout(graph: nx.Graph):
    pos = {}
    init_x, init_y = 0.5, 1
    x, y = init_x, init_y
    count = 0
    for node in graph.nodes:
        pos[node] = (x, y)
        y -= 0.1
        x += 0.1
        count += 1
        if count >= np.sqrt(len(graph.nodes)):
            count = 0
            init_y -= 0.1
            init_x -= 0.1
            x, y = init_x, init_y
    return pos


def array_layout(graph: nx.Graph):
    pos = {}
    step = 1 / np.sqrt(len(graph.nodes))
    x_pos = 0
    y_pos = 1
    for node in graph.nodes:
        pos[node] = (x_pos, y_pos)
        x_pos += step
        if x_pos >= 1:
            y_pos -= step
            x_pos = 0
    return pos


def construct_hex_board(board_size):
    """
    Constructs the initial hex board comprising a set of hex cells.
    Each cell has a state and a set of neighbouring cells which only need
    to be computed once.

    :param board_size: size of the board
    :return: returns an array containing HexCells
    """
    board = []
    for i in range(board_size):
        board_row = []
        for j in range(board_size):
            cell_neighbours = get_cell_neighbours(i, j, board_size)
            board_row.append(HexCell(cell_neighbours))
        board.append(board_row)
    return np.array(board)


def get_cell_neighbours(row, col, board_size):
    """
    Computes a list of neighbouring cells for the cell at the given row and column.

    :param row: Row of the hex cell
    :param col: Column of the hex cell
    :param board_size: board size
    :return: list of tuples representing coordinates of neighbouring cells
    """
    ignore = {(row, col), (row - 1, col - 1), (row + 1, col + 1)}
    neighbours = []
    for i in range(row - 1, row + 2):
        for j in range(col - 1, col + 2):
            if (i, j) not in ignore and 0 <= i < board_size and 0 <= j < board_size:
                neighbours.append((i, j))
    return neighbours


if __name__ == '__main__':
    h = Hex(5)
    h.init_game()
    while not h.is_current_state_terminal():
        actions = h.get_legal_actions()
        action_idx = np.random.choice(np.arange(0, len(actions)))
        action = actions[action_idx]
        h.get_child_state(action)
    print("Player {} won.".format("one" if h.state == HexState.PLAYER_ONE_WON else "two"))
    h.visualize()
