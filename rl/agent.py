import os
from rl.mcts import MCTS
from keras.models import Sequential, Model
from keras.layers import Dense, Softmax
from keras.callbacks import Callback
from tensorflow import keras
from tqdm import tqdm
from game.interface import Game
import numpy as np
import tensorflow as tf
from collections import deque

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
        self.mcts_config = config.get("mcts", {})
        self.episodes = config.get("episodes", 50)
        self.rbuf_size = config.get("rbuf_size", 1000)
        self.save_interval = config.get("save_interval", 10)
        self.num_sim = config.get("num_sim", 50)
        self.epochs = anet_config.get("epochs", 10)
        self.batch_size = anet_config.get("batch_size", 100)
        self.file_path = anet_config.get("file_path", f"{ROOT_DIR}/rl/models")
        self.game = game
        self.anet = ANET(
            hidden_layers=anet_config.get("hidden_layers", [(32, "relu")]),
            output_nodes=game.number_of_actions(),
            optimizer=anet_config.get("optimizer", "adam"),
            learning_rate=anet_config.get("learning_rate", 0.01),
            weight_file=anet_config.get("weight_file", None)
        )
        if config.get('default_policy') == 'uniform':
            self.mcts_default_policy = lambda _, legal_actions: np.random.choice(legal_actions)
        elif config.get('default_policy') == 'agent':
            self.mcts_default_policy = lambda state, legal_actions: self.propose_action(state, legal_actions)
        else:
            raise ValueError('invalid default_policy conig')

        self.mcts_tree = None

    def train(self):
        """
        Learns an action policy by utilising monte carlo tree search simulations
        """
        rbuf_x = deque(maxlen=self.rbuf_size)
        rbuf_y = deque(maxlen=self.rbuf_size)
        progress = tqdm(range(self.episodes), desc="Episode")
        for episode in progress:
            # reset mcts tree
            self.mcts_tree = MCTS(config=self.mcts_config, game=self.game, default_policy=self.mcts_default_policy)

            state = self.game.get_initial_state()
            while not self.game.is_state_terminal(state):
                distribution = self.mcts_tree.simulate(state=state, num_sim=self.num_sim)
                rbuf_x.append(state)
                rbuf_y.append(distribution)
                action_idx = np.argmax(distribution)
                state = self.game.get_child_state(state, action_idx)
                self.mcts_tree.retain_subtree(action_idx)

            history = LossHistory()
            self.anet.fit(x=np.array(rbuf_x), y=np.array(rbuf_y), batch_size=self.batch_size, epochs=self.epochs,
                          verbose=3, callbacks=[history])
            if episode % self.save_interval == 0:
                self.anet.save_weights(filepath=f"{self.file_path}/anet_episode_{episode}")

            progress.set_description(
                "Batch loss: {:.4f}".format(history.losses[-1]) +
                " | Average loss: {:.4f}".format(np.mean(history.losses)) +
                " | RBUF Size: {}".format(len(rbuf_x))
            )

    def propose_action(self, state, actions):
        """
        Proposes an action for the given state

        :param state: state of the game
        :param actions: Possible actions from the given state (not all actions)
        :return: optimal action according to the learned policy
        """
        tensor = tf.convert_to_tensor(np.array([state]))
        distribution = self.anet(tensor)[0]
        distribution = np.array(distribution)

        all_actions_idx = np.arange(0, self.game.number_of_actions())
        illegal_actions_idx = np.setdiff1d(all_actions_idx, actions)
        distribution[illegal_actions_idx] = 0.0
        distribution = distribution / np.sum(distribution)

        action_idx = np.argmax(distribution)
        return action_idx


class LossHistory(Callback):

    def __init__(self):
        super().__init__()
        self.losses = []

    def on_train_begin(self, logs=None):
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))


def configure_optimizer(optimizer, learning_rate):
    if optimizer == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate)
    if optimizer == "rms_prop":
        return keras.optimizers.RMSprop(learning_rate=learning_rate)
    if optimizer == "adagrad":
        return keras.optimizers.Adagrad(learning_rate=learning_rate)
    return keras.optimizers.Adam(learning_rate=learning_rate)


class ANET(Model):

    def __init__(self, hidden_layers, output_nodes, optimizer, learning_rate, weight_file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layers = [Dense(nodes, activation=activation) for nodes, activation in hidden_layers]
        layers.append(Dense(output_nodes))
        layers.append(Softmax())
        self.model = Sequential(layers)
        self.compile(
            optimizer=configure_optimizer(optimizer, learning_rate),
            loss=keras.losses.kl_divergence,
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
