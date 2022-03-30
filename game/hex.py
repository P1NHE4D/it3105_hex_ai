import dataclasses
from enum import Enum
from typing import Any

from game.interface import Game
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


@dataclasses.dataclass
class HexBoardState:
    player_to_move: int
    # np.array of np.array of HexCellState
    board: Any


class HexCellState(Enum):
    EMPTY = 0
    PLAYER_ONE = 1
    PLAYER_TWO = 2


class HexWinState(Enum):
    UNDECIDED = 0
    PLAYER_ONE_WON = 1
    PLAYER_TWO_WON = 2


class HexCell:

    def __init__(self, neighbours):
        self.neighbours = neighbours
        self.state = HexCellState.EMPTY


def matrix_cells(matrix):
    """
    Yield each cell of matrix, along with it's row and col (such that matrix[row, col] == cel)
    """
    for row, elements in enumerate(matrix):
        for col, cell in enumerate(elements):
            yield row, col, cell


class Hex(Game):

    def get_initial_state(self):
        """
        Returns a one-hot encoded representation of the initial state

        One hot encoding representation:
        - first two digits indicate current player: [10...] player one, [01....] player two
        - remaining digits encode each cell state with 2 bits per cell: [...00...] empty,
          [...10...] player one, [...01...] player two
        - length of ohe vector is 2 + 2 * board_size * board_size

        :return: one-hot encoded representation of the initial state
        """
        return self._encode_state(HexBoardState(
            player_to_move=0,
            board=construct_hex_board(self.board_size),
        ))

    def is_state_terminal(self, state):
        # we can do this because Hex doesn't have ties
        return self.get_state_reward(state) != 0

    def player_to_move(self, state):
        return self._decode_state(state).player_to_move

    def next_player_to_move(self, state):
        return (self._decode_state(state).player_to_move + 1) % 2

    def number_of_actions(self):
        return len(self.all_actions)

    def _encode_state(self, state: HexBoardState):
        ohe_state = np.zeros(2 + 2 * self.board_size ** 2)
        ohe_state[state.player_to_move] = 1
        for row, col, cell in matrix_cells(state.board):
            if cell.state != HexCellState.EMPTY:
                player_offset = 0 if cell.state == HexCellState.PLAYER_ONE else 1
                index = 2 + 2 * (row * self.board_size + col) + player_offset
                ohe_state[index] = 1
        return ohe_state

    def _decode_state(self, ohe_state):
        # != as xor
        if not (bool(ohe_state[0]) != bool(ohe_state[1])):
            raise ValueError('hex state encoding erroneous player-to-move')
        player_to_move = 0 if ohe_state[0] == 1 else 1
        board = construct_hex_board(self.board_size)
        for row, col, cell in matrix_cells(board):
            idx_player_0 = 2 + 2 * (row * self.board_size + col)
            idx_player_1 = idx_player_0 + 1
            if ohe_state[idx_player_0] == 1 and ohe_state[idx_player_1] == 1:
                raise ValueError('hex state encoding both players cannot occupy the same cell')
            if ohe_state[idx_player_0] == 1:
                cell.state = HexCellState.PLAYER_ONE
            elif ohe_state[idx_player_1] == 1:
                cell.state = HexCellState.PLAYER_TWO
        return HexBoardState(
            player_to_move=player_to_move,
            board=board,
        )

    def __init__(self, board_size):
        super().__init__()
        self.board_size = board_size
        self.all_actions = [(row, col) for row in range(self.board_size) for col in range(self.board_size)]

    def get_legal_actions(self, state):
        """
        :return: returns array comprising possible actions in current state
        """
        decoded = self._decode_state(state)
        return [
            i for i, rowcol in enumerate(self.all_actions)
            if decoded.board[rowcol[0]][rowcol[1]].state == HexCellState.EMPTY
        ]

    def get_action(self, index):
        return self.all_actions[index]

    def get_child_state(self, state, action):
        """
        Determines the child state based on the chosen action, returning
        a one-hot encoded successor state.

        :param action: picked action
        :return: one-hot encoded successor state
        """
        decoded = self._decode_state(state)
        row, col = self.all_actions[action]
        decoded.board[
            row, col].state = HexCellState.PLAYER_ONE if decoded.player_to_move == 0 else HexCellState.PLAYER_TWO
        decoded.player_to_move = (decoded.player_to_move + 1) % 2
        return self._encode_state(decoded)

    def connected_lines(self, state, row, col, visited=None):
        """
        Computes the number of connected lines (rows if player 1, columns if player 2). For example: if this returns 2
        when the cell at row,col is player 1, it means that there is a connection between 2 rows involving row,col.

        Should only be called for non-empty row,col, as an empty cell will result in a noop, and a returned 1 (empty
        cells are not connected to each other in this context)

        :param row: row of cell to search from
        :param col: col of cell to search from
        :param visited: visited nodes that will be excluded from search
        :return: returns a set containing the indices of the connected lines, along with a set containing the (row, col)
                 visited during the search
        """
        if visited is None:
            visited = set()
        lines = set()
        cell_val = state.board[row, col].state
        if cell_val == HexCellState.EMPTY:
            raise ValueError('calling connected_lines on an empty cell never makes sense. Programmer error')
        elif cell_val == HexCellState.PLAYER_ONE:
            lines.add(row)
        elif cell_val == HexCellState.PLAYER_TWO:
            lines.add(col)
        else:
            raise ValueError('Unknown cell state encountered in connected_lines')

        visited.add((row, col))
        for neighbour in state.board[row, col].neighbours:
            neighbour_row, neighbour_col = neighbour
            if state.board[neighbour_row, neighbour_col].state == cell_val and (
                    neighbour_row, neighbour_col) not in visited:
                neighbour_connected_lines, neighbour_visited = self.connected_lines(state, neighbour_row, neighbour_col,
                                                                                    visited)
                lines = lines.union(neighbour_connected_lines)
                visited = visited.union(neighbour_visited)

        return lines, visited

    def get_state_reward(self, state):
        decoded = self._decode_state(state)
        visited = set()
        for row, col, cell in matrix_cells(decoded.board):
            if cell.state != HexCellState.EMPTY:
                # given a non-empty cell on the board, count how close the connected subgraph containing that cell is
                # to forming a line spanning every row or column, depending on player color inhabited by the cell.
                # Retain cells visited during this search, so we can skip them in later iterations (their subgraph
                # has already been searched) Performance note: there's no need to run this on inner cells of the
                # board, but due to the memoization of visited cells the performance difference should be totally
                # negligible
                num_connected, cell_visited = self.connected_lines(decoded, row, col, visited=visited)
                visited = visited.union(cell_visited)
                if len(num_connected) == self.board_size:
                    return 1 if decoded.player_to_move == 1 else -1
        # Hex cannot end in a tie, so getting here must mean the game is not over yet
        return 0

    def visualize(self, state):
        board = self._decode_state(state).board
        g = nx.Graph()
        for row, hex_row in enumerate(board):
            for col, node in enumerate(hex_row):
                node_color = "white"
                node_edge_color = "black"
                if node.state == HexCellState.PLAYER_ONE:
                    node_color = "red"
                    node_edge_color = "red"
                elif node.state == HexCellState.PLAYER_TWO:
                    node_color = "black"
                g.add_node("node_{}_{}".format(row, col), color=node_color, edge_color=node_edge_color)

        # cannot be integrated into the previous loop as this will mix up the order of the nodes
        for row, hex_row in enumerate(board):
            for col, node in enumerate(hex_row):
                for neighbour in node.neighbours:
                    edge_color = "black"
                    edge_weight = 1
                    if node.state == board[neighbour].state:
                        if node.state == HexCellState.PLAYER_ONE:
                            edge_color = "red"
                            edge_weight = 4
                        elif node.state == HexCellState.PLAYER_TWO:
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
    h = Hex(8)
    state = h.get_initial_state()
    while not h.is_state_terminal(state):
        actions = h.get_legal_actions(state)
        action_idx = np.random.choice(np.arange(0, len(actions)))
        action = actions[action_idx]
        state = h.get_child_state(state, action)
    h.visualize(state)
    print("Player {} won.".format("one" if h.get_state_reward(state) == 1 else "two"))
