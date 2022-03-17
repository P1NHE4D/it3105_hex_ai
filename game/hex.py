from enum import Enum

from game.inferface import Game
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class HexCellState(Enum):
    EMPTY = 0
    PLAYER_ONE = 1
    PLAYER_TWO = 2


class HexCell:

    def __init__(self, neighbours):
        self.neighbours = neighbours
        self.state = HexCellState.EMPTY


class Hex(Game):

    def __init__(self, board_size):
        super().__init__()
        self.board = construct_hex_board(board_size)

    def init_game(self):
        for node in self.board.flatten():
            node.state = HexCellState.EMPTY

    def is_current_state_terminal(self):
        pass

    def get_actions(self):
        pass

    def get_child_state(self, action):
        pass

    def get_state_reward(self):
        pass

    def player_to_move(self):
        pass

    def visualize(self):
        g = nx.Graph()
        node_colors = []
        for row, hex_row in enumerate(self.board):
            for col, node in enumerate(hex_row):
                g.add_node("node_{}_{}".format(row, col))
                node_colors.append("red" if node.state == HexCellState.PLAYER_ONE else "black" if node.state == HexCellState.PLAYER_TWO else "white")
                for neighbour in node.neighbours:
                    g.add_edge("node_{}_{}".format(row, col), "node_{}_{}".format(neighbour[0], neighbour[1]))
        nx.draw(g, edgecolors="black", node_color=node_colors)
        plt.show()


def construct_hex_board(board_size):
    board = []
    for i in range(board_size):
        board_row = []
        for j in range(board_size):
            cell_neighbours = get_cell_neighbours(i, j, board_size)
            board_row.append(HexCell(cell_neighbours))
        board.append(board_row)
    return np.array(board)


def get_cell_neighbours(row, col, board_size):
    ignore = {(row, col), (row - 1, col - 1), (row + 1, col + 1)}
    neighbours = []
    for i in range(row - 1, row + 2):
        for j in range(col - 1, col + 2):
            if (i, j) not in ignore and 0 <= i < board_size and 0 <= j < board_size:
                neighbours.append((i, j))
    return neighbours
