from rl.mcts import MCTS
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.activations import sigmoid, leaky_relu
from keras.callbacks import Callback
from tensorflow import keras
from tqdm import tqdm
from game.inferface import Game
import numpy as np


class Agent:

    def __init__(
            self,
            anet_layer_nodes,
            episodes=50,
            save_interval=10,
            anet_learning_rate=0.01,
            anet_weight_file=None,
            num_sim=50
    ):
        self.episodes = episodes
        self.save_interval = save_interval
        self.anet = ANET(anet_layer_nodes, anet_learning_rate, anet_weight_file)
        self.weight_file = anet_weight_file
        self.num_sim = num_sim
        self.mcts_tree = MCTS()

    def train(self, game: Game):
        rbuf_x = []
        rbuf_y = []
        progress = tqdm(range(self.episodes), desc="Episode")
        for episode in progress:
            state, actions = game.init_game()
            while not game.is_current_state_terminal():
                distribution = self.mcts_tree.simulate(game=game, num_sim=self.num_sim)
                rbuf_x.append(state)
                rbuf_y.append(distribution)
                action = actions[np.argmax(distribution)]
                state, actions = game.get_child_state(action)
                self.mcts_tree.retain_subtree(action)
            # TODO: set batch size and number of epochs
            history = LossHistory()
            self.anet.fit(x=rbuf_x, y=rbuf_y, verbose=3, callbacks=[history])
            if episode % self.save_interval == 0:
                # TODO: add parameter for filepath
                self.anet.save_weights(filepath=f"models/anet_episode_{episode}")
            progress.set_description(
                "Batch loss: {:.2f}".format(history.losses[-1]) +
                " | Average loss: {:.2f}".format(np.mean(history.losses))
            )

    def propose_action(self, state, actions):
        distribution = self.anet.predict([state])[0]
        action = actions[np.argmax(distribution)]
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

    def __init__(self, layer_nodes, learning_rate, weight_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = [Dense(nodes, activation=leaky_relu) for nodes in layer_nodes[:-1]]
        layers.append(Dense(layer_nodes[-1], activation=sigmoid))
        self.model = Sequential(layers)
        self.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.binary_crossentropy
        )
        try:
            self.load_weights(weight_file)
        except:
            print("Unable to load weight file")

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def get_config(self):
        super().get_config()
