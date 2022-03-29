from oht.ActorClient import ActorClient
from rl.agent import Agent


class OHTClient(ActorClient):

    def __init__(self, agent: Agent):
        super(OHTClient, self).__init__()
        self.agent = agent

    def handle_get_action(self, state):
        pass
