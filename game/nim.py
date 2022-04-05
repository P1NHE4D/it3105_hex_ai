import dataclasses
import numpy as np
from game.interface import Game


class Nim(Game):

    def __init__(self, stones=4, max_take=2, **kwargs):
        """
        Nim is a game where two player alternate taking stones from a central pile. There are 'stones' stones,
        and a player can take a number of stones on their turn in the range [1,max_take]. The player who takes the last
        stone wins.

        State representation is a 'stones'+2 length tuple containing for the i+2'th stone 1 if the stone has been taken,
        and 0 otherwise. The first two positions describe the player to move: (1 0) if it's player 0's turn, (0 1) if
        it's player 1's turn. The stones are taken in the order of left to right. For example, a 'stones=4' game would
        start as (1 0 1 1 1 1), and become (0 1 0 1 1 1) after player 0 takes one stone.

        Player 0 and 1 receive 1.0 and -1.0 when winning, respectively. 0 wants to maximize, 1 wants to minimize

        :param stones: number of stones on the board
        :param max_take: maximum number of stones a player can take on their turn
        :return:
        """
        super().__init__(**kwargs)
        self.stones = stones
        self.max_take = max_take

        # materialized list of all possible actions, to be filtered and indexed into
        self.actions = list(np.arange(1, self.max_take + 1))

        # state of the game described through the number of remaining stones, as well as the ohe of the current player
        # + the number of remaining stones. To be set by 'get_initial_state'
        self.remaining: int = None
        self.ohe_state: np.array = None

    def get_initial_state(self):
        # reset state
        self.current_player = self.starting_player
        self.remaining = self.stones
        self.ohe_state = np.ones(2 + self.stones)
        self.ohe_state[:2] = 0
        self.ohe_state[self.current_player] = 1

        return self.ohe_state

    def get_current_state(self):
        return self.ohe_state

    def number_of_actions(self):
        return self.max_take

    def get_action(self, index):
        return self.actions[index]

    def is_state_terminal(self):
        return self.remaining == 0

    def get_legal_actions(self):
        return [i for i, take in enumerate(self.actions) if take <= self.remaining]

    def get_child_state(self, index):
        take = self.get_action(index)
        # update state
        self.current_player = (self.current_player + 1) % 2
        self.remaining -= take

        took = take

        # update ohe representation
        stones_taken = self.stones - self.remaining
        self.ohe_state[2 + stones_taken - took + np.arange(took)] = 0

        return self.ohe_state

    def get_state_reward(self):
        if not self.is_state_terminal():
            return 0.0

        # notice in Nim a draw is not possible
        # the winner is the player whose turn it is not
        if self.current_player == 1:
            return 1.0
        elif self.current_player == 0:
            return -1.0

    def next_player_to_move(self):
        return (self.current_player + 1) % 2

    def visualize(self):
        print(f"PLAYER {self.current_player} STONES {self.stones} MAX_TAKE {self.max_take} "
              f"REMAINING {self.remaining} OHE {self.ohe_state}")


if __name__ == '__main__':
    g = Nim(stones=6, max_take=2)
    print("solving game with random walk")
    state = g.get_initial_state()
    print("initial state", state)
    print("first player is", g.player_to_move())
    while not g.is_state_terminal():
        print("visualizing current state...")
        g.visualize()
        print("player to move", g.player_to_move())
        a = g.get_legal_actions()
        print("actions", a)
        if g.is_state_terminal():
            break
        to_take = a[np.random.choice(np.arange(len(a)))]
        print("taking action", to_take)
        state = g.get_child_state(to_take)
        print("resulted in state", state)
    print("reward", g.get_state_reward())
