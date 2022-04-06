import os

from rl.agent import Agent
from rl.mcts import MCTS
from tqdm import tqdm
from game.interface import Game
import numpy as np
from collections import deque

from rl.nn import LossHistory, ANET, Critic, LiteModel

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class ANETAgent(Agent):

    def __init__(
            self,
            game: Game,
            config=None
    ):
        """
        Agent which chooses its action based on a trained ANET

        :param game: game instance
        :param config: agent config
        """
        if config is None:
            config = {}
        anet_config = config.get("anet", {})
        self.mcts_config = config.get("mcts", {})
        self.episodes = config.get("episodes", 50)
        self.rbuf_size = config.get("rbuf_size", 1000)
        self.save_interval = config.get("save_interval", 10)
        self.num_sim = config.get("num_sim", 50)
        self.min_sim = config.get("min_sim", 50)
        self.epochs = anet_config.get("epochs", 10)
        self.batch_size = anet_config.get("batch_size", 100)
        self.file_path = anet_config.get("file_path", f"{ROOT_DIR}/rl/models")
        self.dynamic_sim = config.get("dynamic_sim", False)
        self.epsilon = config.get("epsilon", 0)
        self.epsilon_decay = config.get("epsilon_decay", 1)
        self.fit_interval = config.get("fit_interval", 1)
        self.game = game
        self.anet = ANET(
            input_shape=self.game.get_initial_state().shape,
            hidden_layers=anet_config.get("hidden_layers", [(32, "relu")]),
            output_nodes=game.number_of_actions(),
            optimizer=anet_config.get("optimizer", "adam"),
            learning_rate=anet_config.get("learning_rate", 0.01),
            weight_file=anet_config.get("weight_file", None)
        )
        self.visualize_episode = config.get('visualize_episode', False)
        self.anet_lite: LiteModel = LiteModel.from_keras_model(self.anet)

        self.sigma = config.get("sigma", 1)
        self.sigma_decay = config.get("sigma_decay", 1)
        self.sigma_delay = config.get("sigma_delay", 20)
        critic_config = config.get("critic", {})
        self.critic = Critic(
            input_shape=self.game.get_initial_state().shape,
            hidden_layers=critic_config.get("hidden_layers", [(32, "relu")]),
            optimizer=critic_config.get("optimizer", "adam"),
            learning_rate=critic_config.get("learning_rate", 0.01),
            weight_file=critic_config.get("weight_file", None)
        )
        self.critic_lite: LiteModel = LiteModel.from_keras_model(self.critic)

        default_policy_str = config.get('default_policy', "uniform")
        if default_policy_str == 'uniform':
            self.mcts_default_policy = lambda _, legal_actions: np.random.choice(legal_actions)
        else:
            self.mcts_default_policy = lambda state, legal_actions: self.propose_action(state, legal_actions)

        self.mcts_tree = None

    def train(self):
        """
        Learns an action policy by utilising monte carlo tree search simulations. Returns a list of weight files saved
        during training
        """
        weight_files = []
        rbuf_x = deque(maxlen=self.rbuf_size)
        rbuf_y = deque(maxlen=self.rbuf_size)
        cbuf_x = []
        cbuf_y = []
        progress = tqdm(range(1, self.episodes + 1), desc="(initial episode)")
        history = LossHistory()

        # save initial weights as episode 0
        weight_file = f"{self.file_path}/anet_episode_0"
        self.anet.save_weights(filepath=weight_file)
        weight_files.append(weight_file)

        for episode in progress:
            # reset mcts tree
            self.mcts_tree = MCTS(
                config=self.mcts_config,
                default_policy=self.mcts_default_policy,
                epsilon=self.epsilon,
                critic=self.critic_lite,
                sigma=self.sigma
            )

            # plays the game using action distributions returned from the mcts tree
            state = self.game.get_initial_state()
            while not self.game.is_state_terminal():

                # compute number of simulations if dynamic simulations is enabled
                num_sim = self.num_sim
                if self.dynamic_sim:
                    actions = self.game.get_legal_actions()
                    factor = 1 - (len(actions) / self.game.number_of_actions())
                    num_sim = max([round(factor * self.num_sim), self.min_sim])

                # obtain an action distribution from mcts
                distribution = self.mcts_tree.simulate(game=self.game, num_sim=num_sim)

                # store the distribution and respective state for training
                rbuf_x.append(np.copy(state))
                rbuf_y.append(distribution)

                # select the best action and advance the game
                action = np.argmax(distribution)
                state = self.game.get_child_state(action)

                # append child state to the critic buffer for training
                cbuf_x.append(np.copy(state))

                self.mcts_tree.retain_subtree(action)

            if self.visualize_episode:
                self.game.visualize()

            cbuf_y.extend(np.full(len(cbuf_x) - len(cbuf_y), fill_value=self.game.get_state_reward()))

            # train ANET and critic
            if episode % self.fit_interval == 0 or episode == self.episodes:
                self.anet.fit(x=np.array(rbuf_x), y=np.array(rbuf_y), batch_size=self.batch_size, epochs=self.epochs,
                              verbose=3, callbacks=[history])
                self.anet_lite = LiteModel.from_keras_model(self.anet)

                # only afford fitting the critic if there is (or there will be) a non-zero chance of it being used
                if self.sigma < 1 or self.sigma_decay < 1:
                    self.critic.fit(x=np.array(cbuf_x), y=np.array(cbuf_y), verbose=3)
                    self.critic_lite = LiteModel.from_keras_model(self.critic)
                cbuf_x = []
                cbuf_y = []

            # store weight files
            if episode % self.save_interval == 0 or episode == self.episodes:
                weight_file = f"{self.file_path}/anet_episode_{episode}"
                self.anet.save_weights(filepath=weight_file)
                weight_files.append(weight_file)

            # update epsilon and sigma
            self.epsilon *= self.epsilon_decay
            if episode >= self.sigma_delay:
                self.sigma *= self.sigma_decay

            progress.set_description(
                "Batch loss: {:.4f}".format(history.losses[-1] if len(history.losses) > 0 else -1) +
                " | Average loss: {:.4f}".format(np.mean(history.losses) if len(history.losses) > 0 else -1) +
                " | RBUF Size: {}".format(len(rbuf_x)) +
                " | Epsilon: {}".format(self.epsilon) +
                " | Sigma: {}".format(self.sigma)
            )

        return weight_files

    def propose_action(self, state, legal_actions):
        """
        Proposes an action for the given state

        :param state: state of the game
        :param legal_actions: Possible actions from the given state (not all actions)
        :return: optimal action according to the learned policy
        """
        distribution = self.anet_lite.predict(np.array([state]))[0]
        distribution = np.array(distribution)

        all_actions_idx = np.arange(0, self.game.number_of_actions())
        illegal_actions_idx = np.setdiff1d(all_actions_idx, legal_actions)
        distribution[illegal_actions_idx] = 0.0
        distribution = distribution / np.sum(distribution)

        action_idx = np.argmax(distribution)
        return action_idx
