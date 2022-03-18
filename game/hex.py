from enum import Enum

from game.inferface import Game
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
        self.actions: list = []
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
            node.state = HexCellState.EMPTY

        # initialize new one hot encoding
        self.ohe_board = np.zeros(2 + 2 * self.board_size ** 2)
        self.ohe_board[self.current_player] = 1

        # reset actions
        self.actions = [(row, col) for row in range(self.board_size) for col in range(self.board_size)]

        # reset game state
        self.state = HexState.UNDECIDED

        return self.ohe_board

    def is_current_state_terminal(self):
        """
        :return: True if terminal, otherwise False
        """
        return self.state == HexState.PLAYER_ONE_WON or self.state == HexState.PLAYER_TWO_WON

    def get_actions(self):
        """
        :return: returns array comprising possible actions in current state
        """
        return self.actions

    def get_child_state(self, action):
        """
        Determines the child state based on the chosen action, returning
        a one-hot encoded successor state.

        :param action: picked action
        :return: one-hot encoded successor state
        """

        # set board state
        row, col = action
        self.board[row, col].state = HexCellState.PLAYER_ONE if self.current_player == 0 else HexCellState.PLAYER_TWO

        # remove picked action from possible actions
        self.actions.remove((row, col))

        self.update_game_state(row, col)

        # update one hot encoding
        ohe_index = 2 + 2 * (row * self.board_size + col)
        if self.current_player == 1:
            ohe_index += 1
        self.ohe_board[ohe_index] = 1
        self.ohe_board[self.current_player] = 0
        self.next_player()
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
                if self.board[neighbour_row, neighbour_col].state == cell_val and (neighbour_row, neighbour_col) not in visited:
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

        # TODO: layout needs to be set to a fixed position. Remove labels once this is done.
        # TODO: highlight connections between adjacent pieces belonging to the same player
        g = nx.Graph()
        node_colors = []
        for row, hex_row in enumerate(self.board):
            for col, node in enumerate(hex_row):
                g.add_node("node_{}_{}".format(row, col))
                node_colors.append(
                    "red" if node.state == HexCellState.PLAYER_ONE else "black" if node.state == HexCellState.PLAYER_TWO else "white")

        # edges cannot be moved in the upper loop as this will mix up the order of the nodes and cause
        # issues with the color map
        for row, hex_row in enumerate(self.board):
            for col, node in enumerate(hex_row):
                for neighbour in node.neighbours:
                    g.add_edge("node_{}_{}".format(row, col), "node_{}_{}".format(neighbour[0], neighbour[1]))
        nx.draw(g, edgecolors="black", node_color=node_colors, with_labels=True)
        plt.show()


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
