import functools
import numpy as np
import gymnasium.spaces as spaces

from pettingzoo import ParallelEnv
from copy import copy
from numpy.random import Generator

from common.agent import Agent

class MarlEnvironment(ParallelEnv):
  metadata = {
    "name": "Marl_Environment",
    "render_fps": 10
  }
  
  def __init__(
      self, 
      mapsize: int=100, 
      max_timesteps: int=1000, 
      agents: list[Agent] = [Agent()]*3,
      num_near_agents: int=2, 
      max_speed: int=5, 
      max_accel: np.float32=1.0, 
      max_angular_accel: np.float32=0.2*np.pi, 
      render_fps: int|None=None, 
      render_mode: bool=True,
      render_vectors: bool=True,
  ):
    
    self.mapsize = mapsize
    self.max_timesteps = max_timesteps
    self.agents_list = agents
    self.num_agents = len(self.agents_list)
    self.max_speed = max_speed
    self.max_accel = max_accel
    self.max_angular_accel = max_angular_accel

    try:
      assert num_near_agents < self.num_agents
    except AssertionError as e:
      raise e(f"num_near_agents must be less than num_agents. You chose num_agents as {num_agents} and num_near_agents as {num_near_agents}")
    
    self.num_near_agents = num_near_agents
    self.render_mode =  render_mode
    self.render_fps = render_fps if render_fps else self.metadata["render_fps"]
    self.render_vectors = render_vectors

    self.agents = [f"agent_{i}" for i in range(self.num_agents)]

    self.targets_x = [0]*self.num_agents
    self.targets_y = [0]*self.num_agents

    # self.agents_x = [0]*self.num_agents
    # self.agents_y = [0]*self.num_agents
    # self.agents_speed = [0]*self.num_agents
    # self.agents_heading = [0]*self.num_agents
    # self.agents_angular_vel = [0]*self.num_agents

    self.timestep = 0
    self.possible_agents = ["agent"]

    self.observation_spaces = {
      "agent": spaces.Dict(
        {
          "heading": spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float32),
          "speed": spaces.Box(low=0, high=self.max_speed, shape=(1,), dtype=np.float32),
          "target_heading": spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float32),
          "target_dist": spaces.Box(low=0, high=self.mapsize, shape=(1,), dtype=np.float32), 
          "nearby_agents": spaces.Dict(
            {
              f"agent_{i}": spaces.Dict(
                {
                  "agent_heading": spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float32),
                  "agent_dist": spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float32)
                }
              ) for i in range(self.num_near_agents)
            }
          )
        }
      )
    }

    self.action_spaces = {
      "agent": spaces.Box(low=(-self.max_angular_accel, -self.max_accel), high=(self.max_angular_accel, self.max_accel), shape=(2,), dtype=np.float32)
    }

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    self.timestep = 0

    for i, agent in enumerate(self.agents_list):
      agent.reset()
      agent.x, agent.y = Generator.random(size=2)*self.mapsize
      self.targets_x[i], self.targets_y[i] = Generator.random(size=2)*self.mapsize

    observations = self._get_obs()
    infos = self._get_infos()

    return observations, infos

  def step(self, actions):
    for agent in self.agents_list:
      action = actions[agent.name]
      agent.angular_vel += action[0]
      agent.speed += action[1]

      agent.heading += agent.angular_vel
      agent.x += agent.speed*np.cos(agent.heading)
      agent.y += agent.speed*np.sin(agent.heading)

    observations = self._get_obs()
    rewards = self._get_rewards(observations, actions)
    terminations = self._get_terms()
    truncations = self._get_truncs()
    infos = self._get_infos()

    self.timestep += 1

    return observations, rewards, terminations, truncations, infos

  def _get_obs(self):
    observations = {agent.name: {} for agent in self.agents_list}
    
    for agent, target in zip(self.agents_list, zip(self.targets_x, self.targets_y)):
      neighbors = [neighbor for neighbor in self.agents_list if neighbor.name != agent.name].sort(key=lambda neighbor: np.sqrt((agent.x-neighbor.x)**2+(agent.y-neighbor.y)**2))
      
      observations["agent.name"].update(
        {
          "heading": agent.heading,
          "speed": agent.speed,
          "target_heading": np.atan2(agent.y-target[1], agent.x-target[0]) - agent.heading,
          "target_dist": np.sqrt((agent.x-target[0])**2+(agent.y-target[1])**2),
          "nearby_agents": {
            f"agent_{i}": {
              "agent_heading": np.atan2(agent.y-neighbors[i].y, agent.x-neighbors[i].x) - agent.heading,
              "agent_dist": np.sqrt((agent.x-neighbors[i].x)**2+(agent.y-neighbors[i].y)**2),
            } for i in range(self.num_near_agents)
          }
        }
      )

    return observations

  def _get_rewards(self, observations, actions):
    r_neighbor_prox = -10
    r_target_prox = 50
    
    rewards = {agent.name: 0 for agent in self.agents_list}

    for agent, target in zip(self.agents_list, zip(self.targets_x, self.targets_y)):
      neighbors = [neighbor for neighbor in self.agents_list if neighbor.name != agent.name].sort(key=lambda neighbor: np.sqrt((agent.x-neighbor.x)**2+(agent.y-neighbor.y)**2))
      for neighbor in neighbors:
        rewards[agent.name] += r_neighbor_prox/np.sqrt((agent.x-neighbor.x)**2+(agent.y-neighbor.y)**2)
        rewards[agent.name] += r_target_prox/np.sqrt((agent.x-target[0])**2+(agent.y-target[1])**2)

  def _get_terms(self):
    pass

  def _get_truncs(self):
    return [False]*self.num_agents if self.timestep < self.max_timesteps else [True]*self.num_agents
  
  def _get_infos(self):
    pass

  def render(self):
    pass

  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
    return self.observation_spaces[agent]
  
  @functools.lru_cache(maxsize=None)
  def action_space(self, agent):
    return self.action_spaces[agent]