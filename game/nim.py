import numpy as np

from game.interface import Game


class Nim(Game):
    def __init__(self, stones=4, max_take=2, **kwargs):
        """
        Nim is a game where two player alternate taking stones from a central pile. There are 'stones' stones,
        and a player can take a number of stones on their turn in the range [1,max_take]. The player who takes the last
        stone wins.

        Representations:
        * State representation is a 'stones' length tuple containing for the i'th stone 1 if the stone has been taken,
          and 0 otherwise. The stones are taken in the order of left to right. For example, a 'stones=4' game would
          start. as [1 1 1 1], and become [0 1 1 1] after a player takes one stone.
        * Action representation is a 'max_take' length OHE tuple containing a 1 in the i'th position, where i+1 is the
          number of stones a player would like to take. For example, a 'max_take=3' game where there are 2 stones left
          would afford the actions: [1 0 0] or [0 1 0], corresponding to "take 1 stone" or "take 2 stones" respectively

        Reward:
        * Player 0 and 1 receive 1.0 and -1.0 when winning, respectively. 0 wants to maximize, 1 wants to minimize

        :param stones: number of stones on the board
        :param max_take: maximum number of stones a player can take on their turn
        :return:
        """
        super().__init__(**kwargs)
        self.stones = stones
        self.max_take = max_take

        # to be set upon first init_game (TODO: why do we have an init game function?)
        self.remaining_stones = None

    def get_current_state(self):
        return self.encode_state()

    def encode_state(self):
        part_player = (float(self.player_to_move()),)
        part_taken = (0.0,) * (self.stones - self.remaining_stones)
        part_untaken = (1.0,) * self.remaining_stones
        return part_player + part_taken + part_untaken

    def init_game(self):
        self.remaining_stones = self.stones
        return self.encode_state()

    def is_current_state_terminal(self):
        return self.remaining_stones == 0

    def get_actions(self):
        actions = []
        for i in range(self.max_take):
            if i+1 > self.remaining_stones:
                # this action (and all actions greater than it) use more stones than are remaining, so break
                break
            actions.append(tuple(1.0 if j == i else 0.0 for j in range(self.max_take)))

        return actions

    def get_child_state(self, action):
        took = np.argmax(action)+1
        self.remaining_stones -= took

        # TODO: should we not advance player if we're in a terminal state?
        self.advance_player()

        return self.encode_state()

    def advance_player(self):
        self.current_player = (self.current_player + 1) % 2

    def get_state_reward(self):
        if not self.is_current_state_terminal():
            # reward ony dealt at terminal state
            # alternatively could treat this as a reward of 0, but for now I'll reserve 0 to mean "draw"
            return None

        # if we're in a terminal state, that means the last player that made a move made the winning move (we have to
        # advance the current player to move because the game automatically moves on to the next player even in a
        # terminal state)
        self.advance_player()

        winner = self.player_to_move()

        # notice in Nim a draw is not possible
        if winner == 0:
            return 1.0
        elif winner == 1:
            return -1.0

    def player_to_move(self):
        return self.current_player

    def visualize(self):
        # extremly bare-bones for now...
        print(f"player to move: {self.player_to_move()}, remaining stones: {self.remaining_stones}")


if __name__ == '__main__':
    g = Nim(stones=5, max_take=3)
    print("solving game with random walk")
    print("initial state", g.init_game())
    print("first player is", g.player_to_move())
    while not g.is_current_state_terminal():
        print("visualizing current state...")
        g.visualize()
        print("player to move", g.player_to_move())
        a = g.get_actions()
        print("actions", a)
        to_take = a[np.random.choice(np.arange(len(a)))]
        print("taking action", to_take)
        print("resulted in state", g.get_child_state(to_take))
    print("reward", g.get_state_reward())