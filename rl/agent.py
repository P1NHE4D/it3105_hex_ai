import os

from game.nim import Nim
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
            anet_config.get("hidden_layer_nodes", [32]),
            game.get_action_length(),
            anet_config.get("learning_rate", 0.01),
            anet_config.get("weight_file", None)
        )
        self.mcts_tree = MCTS()

    def train(self):
        rbuf_x = []
        rbuf_y = []
        progress = tqdm(range(self.episodes), desc="Episode")
        for episode in progress:
            # reset mcts tree
            self.mcts_tree = MCTS()

            state = self.game.init_game()
            actions = self.game.get_actions()
            while not self.game.is_current_state_terminal():
                distribution = self.mcts_tree.simulate(game=self.game, num_sim=self.num_sim)
                rbuf_x.append(state)
                rbuf_y.append(distribution)
                action = actions[np.argmax(distribution)]
                state = self.game.get_child_state(action)
                actions = self.game.get_actions()
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
        distribution = self.anet.predict([state])[0]

        for i in range(len(distribution)):
            for action in actions:
                if action[i] == 1.0:
                    # if there exists a legal action corresponding to this position of the distribution, do nothing
                    break
            else:
                # otherwise, don't allow this action
                distribution[i] = 0.0

        # rescale to sum to 1
        distribution = distribution / np.sum(distribution)

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
        except Exception as e:
            print("Unable to load weight file", e)

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs)

    def fit(self, **kwargs):
        res = super().fit(**kwargs)
        self.model_trained = True
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
        if not self.model_trained:
            raise Exception("Model needs to be trained first")
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

if __name__ == '__main__':
    nim_config = {'stones': 13, 'max_take': 3}
    game = Nim(**nim_config)
    a = Agent(game, config={
        'num_sim': 500,
        'anet': {
            'weight_file': 'rl/models/anet_episode_40',
        }
    })
    #a.train()

    while True:
        game = Nim(**nim_config)
        print("initial state", game.init_game())
        while not game.is_current_state_terminal():
            print("computer to move")
            chosen = a.propose_action(game.get_current_state(), game.get_actions())
            print("it chose", chosen)
            game.get_child_state(chosen)
            print("state is now", game.get_current_state())
            if game.is_current_state_terminal():
                print("game end")
                break
            actions = game.get_actions()
            i = int(input(f"your move ({[(i, a) for i, a in enumerate(actions)]}):"))
            print("you chose", actions[i])
            game.get_child_state(actions[i])
            print("state is now", game.get_current_state())
        print("winner was", game.get_state_reward())
