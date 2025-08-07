import functools
import pygame
import numpy as np
import gymnasium.spaces as spaces

from pettingzoo import ParallelEnv
from copy import copy
from numpy.random import Generator

from common.agent import Agent

class MarlEnvironment(ParallelEnv):
  metadata = {
    "name": "Marl_Environment",
    "render_modes": ["human", "rgb_array"],
    "render_fps": 10,
    "agent_radius": 0.5,
    "goal_radius": 2.5
  }
  
  def __init__(
      self, 
      mapsize: int=100, 
      max_timesteps: int=1000, 
      agents: list[Agent] = [Agent()]*3,
      num_near_agents: int=2, 
      max_speed: int=5, 
      max_angular_speed: np.float32=10*np.pi,
      max_accel: np.float32=1.0, 
      max_angular_accel: np.float32=1*np.pi, 
      render_fps: int|None=None, 
      render_mode: str="human",
      render_vectors: bool=True,
  ):
    self.window_size = 512
    self.mapsize = mapsize
    self.max_timesteps = max_timesteps
    self.agents_list = agents
    self.num_agents = len(self.agents_list)
    self.max_speed = max_speed
    self.max_angular_speed = max_angular_speed
    self.max_accel = max_accel
    self.max_angular_accel = max_angular_accel

    try:
      assert num_near_agents < self.num_agents
    except AssertionError as e:
      raise e(f"num_near_agents must be less than num_agents. You chose num_agents as {num_agents} and num_near_agents as {num_near_agents}")
    
    self.num_near_agents = num_near_agents
    self.render_mode =  render_mode

    try: 
      assert self.render_mode in self.metadata["render_modes"]
    except AssertionError as e:
      raise e(f"Invalid render mode {self.render_mode}. Choose from {self.metadata["render_modes"]}")

    self.render_fps = render_fps if render_fps else self.metadata["render_fps"]
    self.render_vectors = render_vectors
    self.window = None
    self.clock = None

    self.agents = [f"agent_{i}" for i in range(self.num_agents)]

    self.targets_x = [0]*self.num_agents
    self.targets_y = [0]*self.num_agents

    # self.agents_x = [0]*self.num_agents
    # self.agents_y = [0]*self.num_agents
    # self.agents_speed = [0]*self.num_agents
    # self.agents_heading = [0]*self.num_agents
    # self.agents_angular_vel = [0]*self.num_agents

    self.terms = None

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
    self.terms = [False]*self.num_agents

    for i, agent in enumerate(self.agents_list):
      agent.reset()
      agent.x, agent.y = Generator.random(size=2)*self.mapsize
      self.targets_x[i], self.targets_y[i] = Generator.random(size=2)*self.mapsize

    if self.render_mode == "human":
      self._render_frame()

    observations = self._get_obs()
    infos = self._get_infos()

    return observations, infos


  def step(self, actions):
    for agent in self.agents_list:
      action = actions[agent.name]
      agent.accel = action[1]
      agent.angular_accel = action[0]
      agent.angular_vel += agent.angular_accel
      agent.angular_vel = np.minimum(self.max_angular_speed, np.maximum(-self.max_angular_speed, agent.angular_vel))
      agent.speed += agent.accel
      agent.speed = np.minimum(self.max_speed, np.maximum(0, agent.speed))

      agent.heading += agent.angular_vel
      agent.heading += 2*np.pi if agent.heading < 0 else -2*np.pi if agent.heading >= 2*np.pi else 0
      agent.x += agent.speed*np.cos(agent.heading)
      agent.x = np.minimum(self.mapsize, np.maximum(0, agent.x))
      agent.y += agent.speed*np.sin(agent.heading)
      agent.y = np.minimum(self.mapsize, np.maximum(0, agent.y))

    if self.render_mode == "human":
      self._render_frame()

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
    r_target_reached = 100
    
    rewards = {agent.name: 0 for agent in self.agents_list}

    for agent, target in zip(self.agents_list, zip(self.targets_x, self.targets_y)):
      neighbors = [neighbor for neighbor in self.agents_list if neighbor.name != agent.name].sort(key=lambda neighbor: np.sqrt((agent.x-neighbor.x)**2+(agent.y-neighbor.y)**2))
      for neighbor in neighbors:
        rewards[agent.name] += r_neighbor_prox/np.sqrt((agent.x-neighbor.x)**2+(agent.y-neighbor.y)**2)
        rewards[agent.name] += r_target_prox/np.sqrt((agent.x-target[0])**2+(agent.y-target[1])**2)
        rewards[agent.name] += r_target_reached if np.sqrt((agent.x-target[0])**2+(agent.y-target[1])**2) < self.metadata["target_radius"] else 0

    return rewards


  def _get_terms(self):
    for i, agent in enumerate(self.agents_list):
      if not self.terms[i]:
        neighbors = [neighbor for neighbor in self.agents_list if neighbor.name != agent.name].sort(key=lambda neighbor: np.sqrt((agent.x-neighbor.x)**2+(agent.y-neighbor.y)**2))
        self.terms[i] = np.sqrt((agent.x-neighbors[0].x)**2+(agent.y-neighbors[0].y)**2) <= 2*self.metadata["agent_radius"] or np.sqrt((agent.x-self.targets_x[i])**2+(agent.y-self.targets_y[i])**2) < self.metadata["target_radius"]

    return self.terms


  def _get_truncs(self):
    return [False]*self.num_agents if self.timestep < self.max_timesteps else [True]*self.num_agents
  

  def _get_infos(self):
    return {agent.name: {} for agent in self.agents_list}


  def render(self):
    if self.render_mode == "rgb_array":
      self._render_frame()


  def _render_frame(self):
    if self.window is None and self.render_mode == "human":
      pygame.init()
      pygame.display.init()
      self.window = pygame.display.set_mode(
        (self.window_size, self.window_size)
      )

    if self.clock is None and self.render_mode == "human":
      self.clock = pygame.time.Clock()

    canvas = pygame.Surface((self.window_size, self.window_size))
    canvas.fill((255, 255, 255))

    pix_size = self.window_size/self.mapsize

    # Draw the agents
    for agent in self.agents_list:
      pygame.draw.circle(
        canvas,
        (0, 0, 255),
        (agent.x, agent.y)*pix_size,
        pix_size*self.metadata["agent_radius"]
      )
      if self.render_vectors:
        # Velocity vector
        pygame.draw.line(
          canvas,
          (255, 255, 0),
          (agent.x, agent.y)*pix_size,
          (agent.x+agent.speed*np.cos(agent.heading), agent.y+agent.speed*np.sin(agent.heading))*pix_size,
          width=3,
        )
        # Acceleration vector
        pygame.draw.line(
          canvas,
          (0, 255, 0),
          (agent.x, agent.y)*pix_size,
          (agent.x+agent.accel*np.cos(agent.heading+agent.angular_accel), agent.y+agent.accel*np.sin(agent.heading+agent.angular_accel))*pix_size,
          width=3
        )

    # Draw the targets
    for target in zip(self.targets_x, self.targets_y):
      pygame.draw.circle(
        canvas,
        (0, 255, 255),
        target*pix_size,
        pix_size*self.metadata["target_radius"]
      )

    if self.render_mode == "human":
      self.window.blit(canvas, canvas.get_rect())
      pygame.event.pump()
      pygame.display.update()
      self.clock.tick(self.metadata["render_fps"])
    else:
      return np.transpose(
        np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
      )


  def close(self):
    if self.window is not None:
      pygame.display.quit()
      pygame.quit()


  @functools.lru_cache(maxsize=None)
  def observation_space(self, agent):
    return self.observation_spaces[agent]
  

  @functools.lru_cache(maxsize=None)
  def action_space(self, agent):
    return self.action_spaces[agent]