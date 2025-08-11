import torch
import numpy as np
from utils import *
from common.agent import Agent

class Actor(Network):
  def __init__(self, input_size: int, output_size: int):
    super().__init__(input_size, 64, output_size)

class Critic(Network):
  def __init__(self, input_size: int):
    super().__init__(input_size, 64, 1)

class HomogMADDPG(Model):
  def __init__(self, agents: list[Agent], actor_dims: list[int], critic_dims: list[int], n_agents: int, n_actions: int):
    self.agents = agents
    self.actor = Actor(*actor_dims, n_actions)
    self.critic = Critic(*critic_dims)
    self.n_agents = n_agents