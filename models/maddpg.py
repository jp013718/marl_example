import numpy as np
from utils import *
from common.agent import Agent


class HomogMADDPG(Model):
  def __init__(self, agents: list[Agent], actor_dims: list[int], critic_dims: list[int], n_agents: int, n_actions: int):
    self.agents = agents
    self.actor = Actor()
    self.critic = Critic()
    self.n_agents = n_agents