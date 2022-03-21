import os
from rl.mcts import MCTS
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.activations import sigmoid, leaky_relu
from keras.callbacks import Callback
from tensorflow import keras
from tqdm import tqdm
from game.interface import Game
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class Agent:

    def __init__(
            self,
            game: Game,
            config=None
    ):
        if config is None:
            config = {}
        anet_config = config.get("anet", {})
        self.episodes = config.get("episodes", 50)
        self.save_interval = config.get("save_interval", 10)
        self.num_sim = config.get("num_sim", 50)
        self.epochs = anet_config.get("epochs", 10)
        self.batch_size = anet_config.get("batch_size", 100)
        self.file_path = anet_config.get("file_path", f"{ROOT_DIR}/rl/models")
        self.game = game
        self.anet = ANET(
            hidden_layer_nodes=anet_config.get("hidden_layer_nodes", [32]),
            output_nodes=game.get_action_length(),
            learning_rate=anet_config.get("learning_rate", 0.01),
            weight_file=anet_config.get("weight_file", None)
        )
        self.mcts_tree = None

    def train(self):
        """
        Learns an action policy by utilising monte carlo tree search simulations
        """
        rbuf_x = []
        rbuf_y = []
        progress = tqdm(range(self.episodes), desc="Episode")
        for episode in progress:
            # reset mcts tree
            self.mcts_tree = MCTS(self)

            state = self.game.init_game()
            while not self.game.is_current_state_terminal():
                distribution = self.mcts_tree.simulate(game=self.game, num_sim=self.num_sim)
                rbuf_x.append(state)
                rbuf_y.append(distribution)
                action_idx = np.argmax(distribution)
                action = self.game.get_action_by_index(action_idx)
                state = self.game.get_child_state(action)
                self.mcts_tree.retain_subtree(action)

            history = LossHistory()
            self.anet.fit(x=np.array(rbuf_x), y=np.array(rbuf_y), batch_size=self.batch_size, epochs=self.epochs, verbose=3, callbacks=[history])
            if episode % self.save_interval == 0:
                self.anet.save_weights(filepath=f"{self.file_path}/anet_episode_{episode}")

            progress.set_description(
                "Batch loss: {:.2f}".format(history.losses[-1]) +
                " | Average loss: {:.2f}".format(np.mean(history.losses))
            )

    def propose_action(self, state, actions):
        """
        Proposes an action for the given state

        :param state: state of the game
        :param actions: Possible actions from the given state (not all actions)
        :return: optimal action according to the learned policy
        """
        distribution = self.anet.predict(np.array([state]))[0]

        all_actions_idx = np.arange(0, len(distribution))
        legal_actions_idx = np.array(list(map(lambda action: self.game.get_action_index(action), actions)))
        illegal_actions_idx = np.setdiff1d(all_actions_idx, legal_actions_idx)
        distribution[illegal_actions_idx] = 0.0
        distribution = distribution / np.sum(distribution)

        action_idx = np.argmax(distribution)
        action = self.game.get_action_by_index(action_idx)
        return action


class LossHistory(Callback):

    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_begin(self, logs=None):
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))


class ANET(Model):

    def __init__(self, hidden_layer_nodes, output_nodes, learning_rate, weight_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = [Dense(nodes, activation=leaky_relu) for nodes in hidden_layer_nodes]
        layers.append(Dense(output_nodes, activation=sigmoid))
        self.model = Sequential(layers)
        self.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.binary_crossentropy
        )
        self.model_trained = False
        try:
            self.load_weights(weight_file)
        except Exception as e:
            print("Unable to load weight file", e)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def fit(self, **kwargs):
        res = super().fit(**kwargs)
        return res

    def predict(
            self,
            x,
            batch_size=None,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
    ):
        return super().predict(
            x,
            batch_size,
            verbose,
            steps,
            callbacks,
            max_queue_size,
            workers,
            use_multiprocessing
        )

    def get_config(self):
        super().get_config()

# if __name__ == '__main__':
#     nim_config = {'stones': 5, 'max_take': 3}
#     game = Nim(**nim_config)
#     a = Agent(game, config={
#         'num_sim': 1000,
#         'anet': {
#             'weight_file': 'rl/models/anet_episode_40',
#         }
#     })
#     a.train()
#
#     wins = 0
#     losses = 0
#     while True:
#         game = Nim(**nim_config)
#         state = game.init_game()
#         while True:
#             chosen = a.propose_action(state, game.get_actions())
#             state = game.get_child_state(chosen)
#             if game.is_current_state_terminal():
#                 break
#             possible_actions = game.get_actions()
#             chosen = possible_actions[np.random.choice(len(possible_actions))]
#             state = game.get_child_state(chosen)
#             if game.is_current_state_terminal():
#                 break
#         reward = game.get_state_reward()
#         if reward == 1.0:
#             wins += 1
#         elif reward == -1.0:
#             losses += 1
#
#         print("wins, losses", wins, losses)
#
#
#
#     while True:
#         game = Nim(**nim_config)
#         print("initial state", game.init_game())
#         while not game.is_current_state_terminal():
#             print("computer to move")
#             chosen = a.propose_action(game.get_current_state(), game.get_actions())
#             print("it chose", chosen)
#             game.get_child_state(chosen)
#             print("state is now", game.get_current_state())
#             if game.is_current_state_terminal():
#                 print("game end")
#                 break
#             actions = game.get_actions()
#             i = int(input(f"your move ({[(i, a) for i, a in enumerate(actions)]}):"))
#             print("you chose", actions[i])
#             game.get_child_state(actions[i])
#             print("state is now", game.get_current_state())
#         print("winner was", game.get_state_reward())
