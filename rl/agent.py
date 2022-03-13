from rl.mcts import MCTS
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.activations import sigmoid, leaky_relu
from keras.callbacks import Callback
from tensorflow import keras
from tqdm import tqdm
from game.inferface import Game
import numpy as np
from main import ROOT_DIR


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
        self.file_path = anet_config.get("file_path", f"{ROOT_DIR}/rl/models")
        self.game = game
        self.anet = ANET(
            anet_config.get("hidden_layer_nodes", [32]),
            len(game.get_actions()),
            anet_config.get("learning_rate", 0.01),
            anet_config.get("weight_file", None)
        )
        self.mcts_tree = MCTS()

    def train(self):
        rbuf_x = []
        rbuf_y = []
        progress = tqdm(range(self.episodes), desc="Episode")
        for episode in progress:
            state, actions = self.game.init_game()
            while not self.game.is_current_state_terminal():
                distribution = self.mcts_tree.simulate(game=self.game, num_sim=self.num_sim)
                rbuf_x.append(state)
                rbuf_y.append(distribution)
                action = actions[np.argmax(distribution)]
                state, actions = self.game.get_child_state(action)
                self.mcts_tree.retain_subtree(action)
            history = LossHistory()
            self.anet.fit(x=rbuf_x, y=rbuf_y, epochs=self.epochs, verbose=3, callbacks=[history])
            if episode % self.save_interval == 0:
                self.anet.save_weights(filepath=f"{self.file_path}/anet_episode_{episode}")
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
            self.model_trained = True
        except:
            print("Unable to load weight file")

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

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
        if not self.model_trained:
            raise Exception("Model needs to be trained first")
        super().predict(
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
