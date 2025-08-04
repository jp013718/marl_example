import functools
import numpy as np
import gymnasium.spaces as spaces

from pettingzoo import ParallelEnv
from copy import copy

class MarlEnvironment(ParallelEnv):
  metadata = {
    "name": "Marl_Environment",
    "render_fps": 10
  }
  
  def __init__(self):
    pass

  def reset(self, seed=None, options=None):
    pass

  def step(self, actions):
    pass

  def _get_obs(self):
    pass

  def _get_infos(self):
    pass

  def _get_rewards(self, observations, actions):
    pass

  def render(self):
    pass

  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
    return self.observation_spaces[agent]
  
  @functools.lru_cache(maxsize=None)
  def action_space(self, agent):
    return self.action_spaces[agent]