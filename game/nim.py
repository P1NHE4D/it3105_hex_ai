import dataclasses

import numpy as np

from game.interface import Game

@dataclasses.dataclass
class NimState:
    player_to_move: int
    remaining_stones: int

class Nim(Game):

    def __init__(self, stones=4, max_take=2, **kwargs):
        """
        Nim is a game where two player alternate taking stones from a central pile. There are 'stones' stones,
        and a player can take a number of stones on their turn in the range [1,max_take]. The player who takes the last
        stone wins.

        State representation is a 'stones' length tuple containing for the i'th stone 1 if the stone has been taken,
        and 0 otherwise. The stones are taken in the order of left to right. For example, a 'stones=4' game would
        start. as (1 1 1 1), and become (0 1 1 1) after a player takes one stone.

        Player 0 and 1 receive 1.0 and -1.0 when winning, respectively. 0 wants to maximize, 1 wants to minimize

        :param stones: number of stones on the board
        :param max_take: maximum number of stones a player can take on their turn
        :return:
        """
        super().__init__(**kwargs)
        self.stones = stones
        self.max_take = max_take

        # materialized list of all possible actions, to be filtered and indexed into
        self.actions = list(np.arange(1, self.max_take+1))

    def get_initial_state(self):
        return self._encode_state(NimState(
            player_to_move=0,
            remaining_stones=self.stones,
        ))

    def number_of_actions(self):
        return self.max_take

    def _encode_state(self, nim_state: NimState):
        encoding = np.ones(1 + self.stones)
        encoding[0] = nim_state.player_to_move
        stones_taken = self.stones - nim_state.remaining_stones
        encoding[np.arange(1, stones_taken + 1)] = 0
        return encoding

    def _decode_state(self, encoded_state) -> NimState:
        player_to_move = encoded_state[0]
        # https://stackoverflow.com/a/25032853
        taken_stones = np.searchsorted(encoded_state[1:], 1)
        remaining_stones = self.stones - taken_stones
        return NimState(
            player_to_move=player_to_move,
            remaining_stones=remaining_stones,
        )

    def is_state_terminal(self, state):
        return self._decode_state(state).remaining_stones == 0

    def get_legal_actions(self, state):
        remaining_stones = self._decode_state(state).remaining_stones
        return [i for i, take in enumerate(self.actions) if take <= remaining_stones]

    def get_child_state(self, encoded_state, action):
        nim_state = self._decode_state(encoded_state)
        take = self.actions[action]
        return self._encode_state(NimState(
            player_to_move=(nim_state.player_to_move + 1) % 2,
            remaining_stones= nim_state.remaining_stones - take,
        ))

    def get_state_reward(self, state):
        if not self.is_state_terminal(state):
            return 0.0
        # the winner is the player who made the move that led to this state
        winner = (self._decode_state(state).player_to_move + 1) % 2

        # notice in Nim a draw is not possible
        if winner == 0:
            return 1.0
        elif winner == 1:
            return -1.0

    def player_to_move(self, encoded_state):
        return self._decode_state(encoded_state).player_to_move

    def next_player_to_move(self, encoded_state):
        return (self._decode_state(encoded_state).player_to_move + 1) % 2

    def visualize(self, encoded_state):
        print(self._decode_state(encoded_state))


if __name__ == '__main__':
    g = Nim(stones=6, max_take=2)
    print("solving game with random walk")
    state = g.get_initial_state()
    print("initial state", state)
    print("first player is", g.player_to_move(state))
    while not g.is_state_terminal(state):
        print("visualizing current state...")
        g.visualize(state)
        print("player to move", g.player_to_move(state))
        a = g.get_legal_actions(state)
        print("actions", a)
        if g.is_state_terminal(state):
            break
        to_take = a[np.random.choice(np.arange(len(a)))]
        print("taking action", to_take)
        state = g.get_child_state(state, to_take)
        print("resulted in state", state)
    print("reward", g.get_state_reward(state))