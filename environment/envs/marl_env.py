import functools
import pygame
import numpy as np
import gymnasium.spaces as spaces

from pettingzoo import ParallelEnv

class MarlEnvironment(ParallelEnv):
  metadata = {
    "name": "Marl_Environment",
    "render_modes": ["human", "rgb_array"],
    "render_fps": 10,
    "agent_radius": 0.5,
    "target_radius": 2.5
  }
  
  def __init__(
      self,
      mapsize: int=100, 
      max_timesteps: int=1000,
      n_agents: int=3, 
      num_near_agents: int=2, 
      max_speed: np.float64=5, 
      max_angular_speed: np.float64=0.5*np.pi,
      max_accel: np.float64=1.0, 
      max_angular_accel: np.float64=0.02*np.pi, 
      render_fps: int|None=None, 
      render_mode: str="human",
      render_vectors: bool=True,
  ):
    self.window_size = 724
    self.mapsize = mapsize
    self.max_timesteps = max_timesteps
    self.max_speed = max_speed
    self.max_angular_speed = max_angular_speed
    self.max_accel = max_accel
    self.max_angular_accel = max_angular_accel
    self.agents = [f"agent_{i}" for i in range(n_agents)]

    try:
      assert num_near_agents < self.num_agents
    except AssertionError as e:
      raise e(f"num_near_agents must be less than num_agents. You chose num_agents as {self.num_agents} and num_near_agents as {num_near_agents}")
    
    self.num_near_agents = num_near_agents
    self.render_mode =  render_mode

    try: 
      assert self.render_mode in self.metadata["render_modes"] or self.render_mode is None
    except AssertionError as e:
      raise e(f'Invalid render mode {self.render_mode}. Choose from {self.metadata["render_modes"]}')

    self.render_fps = render_fps if render_fps else self.metadata["render_fps"]
    self.render_vectors = render_vectors
    self.window = None
    self.clock = None

    self.targets_x = [0]*self.num_agents
    self.targets_y = [0]*self.num_agents

    self.agents_x = [0]*self.num_agents
    self.agents_y = [0]*self.num_agents
    self.agents_speed = [0]*self.num_agents
    self.agents_heading = [0]*self.num_agents
    self.agents_angular_vel = [0]*self.num_agents
    self.agents_angular_accel = [0]*self.num_agents
    self.agents_accel = [0]*self.num_agents

    self.terms = None

    self.timestep = 0
    self.possible_agents = ["agent"]

    self.observation_spaces = {
      "agent": spaces.Dict(
        {
          # "x": spaces.Box(low=0, high=self.mapsize, shape=(1,), dtype=np.float64),
          # "y": spaces.Box(low=0, high=self.mapsize, shape=(1,), dtype=np.float64),
          # "heading": spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float64),
          "speed": spaces.Box(low=0, high=self.max_speed, shape=(1,), dtype=np.float64),
          "angular_vel": spaces.Box(low=-self.max_angular_speed, high=self.max_angular_speed, shape=(1,), dtype=np.float64),
          "target_heading": spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float64),
          "target_dist": spaces.Box(low=0, high=np.sqrt(2)*self.mapsize, shape=(1,), dtype=np.float64), 
          "nearby_agents": spaces.Dict(
            {
              f"agent_{i}": spaces.Dict(
                {
                  "direction_to_agent": spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float64),
                  "agent_dist": spaces.Box(low=0, high=np.sqrt(2)*self.mapsize, shape=(1,), dtype=np.float64),
                  "agent_heading": spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float64),
                  "agent_speed": spaces.Box(low=0, high=self.max_speed, shape=(1,), dtype=np.float64)
                }
              ) for i in range(self.num_near_agents)
            }
          )
        }
      )
    }

    self.action_spaces = {
      "agent": spaces.Box(low=np.array([-self.max_angular_accel, -self.max_accel]), high=np.array([self.max_angular_accel, self.max_accel]), shape=(2,), dtype=np.float64)
    }


  def reset(self, seed=None, options=None):
    self.timestep = 0
    self.terms = {agent: False for agent in self.agents}

    for i in range(len(self.agents)):
      self.agents_heading[i] = 0
      self.agents_speed[i] = 0
      self.agents_angular_vel[i] = 0
  
      self.agents_x[i], self.agents_y[i] = np.random.random(size=2)*self.mapsize
      self.targets_x[i], self.targets_y[i] = np.random.random(size=2)*self.mapsize

    if self.render_mode == "human":
      self._render_frame()

    observations = self._get_obs()
    infos = self._get_infos()

    return observations, infos


  def step(self, actions):
    for i, agent in enumerate(self.agents):
      action = actions[agent]
      
      self.agents_angular_accel[i] = action[0]
      self.agents_accel[i] = action[1]

      self.agents_angular_vel[i] += self.agents_angular_accel[i]
      self.agents_angular_vel[i] = np.minimum(self.max_angular_speed, np.maximum(-self.max_angular_speed, self.agents_angular_vel[i]))
      self.agents_speed[i] += self.agents_accel[i]
      self.agents_speed[i] = np.minimum(self.max_speed, np.maximum(0, self.agents_speed[i]))

      self.agents_heading[i] += self.agents_angular_vel[i]
      self.agents_heading[i] = self.agents_heading[i]%(2*np.pi)
      self.agents_x[i] += self.agents_speed[i]*np.cos(self.agents_heading[i])
      self.agents_x[i] = np.minimum(self.mapsize, np.maximum(0, self.agents_x[i]))
      self.agents_y[i] += self.agents_speed[i]*np.sin(self.agents_heading[i])
      self.agents_y[i] = np.minimum(self.mapsize, np.maximum(0, self.agents_y[i]))

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
    observations = {agent: {} for agent in self.agents}
    
    for i, target_pos in enumerate(zip(self.targets_x, self.targets_y)):
      neighbors = self._get_neighbors(i)

      observations[self.agents[i]].update(
        {
          # "x": self.agents_x[i],
          # "y": self.agents_y[i],
          # "heading": self.agents_heading[i],
          "speed": self.agents_speed[i],
          "angular_vel": self.agents_angular_vel[i],
          "target_heading": (np.atan2(target_pos[1]-self.agents_y[i], target_pos[0]-self.agents_x[i]) - self.agents_heading[i])%(2*np.pi),
          "target_dist": np.sqrt((self.agents_x[i]-target_pos[0])**2+(self.agents_y[i]-target_pos[1])**2),
          "nearby_agents": {
            f"agent_{j}": {
              "direction_to_agent": (np.atan2(self.agents_y[neighbors[j]]-self.agents_y[i], self.agents_x[neighbors[j]]-self.agents_x[i]) - self.agents_heading[i])%(2*np.pi),
              "agent_dist": np.sqrt((self.agents_x[i]-self.agents_x[neighbors[j]])**2+(self.agents_y[i]-self.agents_y[neighbors[j]])**2),
              "agent_heading": (self.agents_heading[neighbors[j]]-self.agents_heading[i])%(2*np.pi),
              "agent_speed": self.agents_speed[neighbors[j]],
            } for j in range(self.num_near_agents)
          }
        }
      )

    return observations


  def _get_rewards(self, observations, actions):
    r_neighbor_prox = -1
    r_neighbor_collision = -1000
    r_target_prox = 10
    r_target_reached = 1.5*r_target_prox*self.max_timesteps
    
    rewards = {agent: 0 for agent in self.agents}

    for i, target_pos in enumerate(zip(self.targets_x, self.targets_y)):
      neighbors = self._get_neighbors(i)
      
      for j in neighbors:
        rewards[self.agents[i]] += r_neighbor_prox/np.sqrt((self.agents_x[i]-self.agents_x[j])**2+(self.agents_y[i]-self.agents_y[j])**2) if np.sqrt((self.agents_x[i]-self.agents_x[j])**2+(self.agents_y[i]-self.agents_y[j])**2) > 0 else 2*r_neighbor_collision
        rewards[self.agents[i]] += r_neighbor_collision if np.sqrt((self.agents_x[i]-self.agents_x[j])**2+(self.agents_y[i]-self.agents_y[j])**2) < 2*self.metadata["agent_radius"] else 0
      
      rewards[self.agents[i]] += r_target_prox/np.sqrt((self.agents_x[i]-target_pos[0])**2+(self.agents_y[i]-target_pos[1])**2)*np.cos(np.atan2(target_pos[1]-self.agents_y[i], target_pos[0]-self.agents_x[i])-self.agents_heading[i])*self.agents_speed[i] if np.sqrt((self.agents_x[i]-target_pos[0])**2+(self.agents_y[i]-target_pos[1])**2) > 0 else 0
      rewards[self.agents[i]] += r_target_reached if np.sqrt((self.agents_x[i]-target_pos[0])**2+(self.agents_y[i]-target_pos[1])**2) < self.metadata["target_radius"] else 0

    return rewards


  def _get_terms(self):
    for i, agent in enumerate(self.agents):
      if not self.terms[agent]:
        neighbors = self._get_neighbors(i)
        if len(neighbors) > 0:
          self.terms[agent] = np.sqrt((self.agents_x[i]-self.agents_x[neighbors[0]])**2+(self.agents_y[i]-self.agents_y[neighbors[0]])**2) <= 2*self.metadata["agent_radius"]
          
        self.terms[agent] = self.terms[agent] or np.sqrt((self.agents_x[i]-self.targets_x[i])**2+(self.agents_y[i]-self.targets_y[i])**2) < self.metadata["target_radius"]

    return self.terms


  def _get_truncs(self):
    return {agent: False if self.timestep < self.max_timesteps else True for agent in self.agents}
  

  def _get_infos(self):
    return {
      agent: {
        "x": self.agents_x[i],
        "y": self.agents_y[i],
        "heading": self.agents_heading[i],
        "speed": self.agents_speed[i],
        "angular_vel": self.agents_angular_vel[i],
        "target_x": self.targets_x[i],
        "target_y": self.targets_y[i],
      } for i, agent in enumerate(self.agents)
    }


  def render(self):
    if self.render_mode == "rgb_array":
      self._render_frame()


  def _get_neighbors(self, agent_idx):
    neighbors = list(range(self.num_agents))
    neighbors.remove(agent_idx)
    neighbors.sort(key=lambda idx: np.sqrt((self.agents_x[agent_idx]-self.agents_x[idx])**2+(self.agents_y[agent_idx]-self.agents_y[idx])**2))

    return neighbors


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

    # Draw the targets
    for target in zip(self.targets_x, self.targets_y):
      pygame.draw.circle(
        canvas,
        (0, 255, 255),
        (target[0]*pix_size, self.mapsize*pix_size-target[1]*pix_size),
        pix_size*self.metadata["target_radius"]
      )

    # Draw the agents
    for i, agent in enumerate(self.agents):
      pygame.draw.circle(
        canvas,
        (0, 0, 255),
        (self.agents_x[i]*pix_size, self.mapsize*pix_size-self.agents_y[i]*pix_size),
        pix_size*self.metadata["agent_radius"]
      )
      if self.render_vectors:
        # Velocity vector
        pygame.draw.line(
          canvas,
          (255, 0, 255),
          (self.agents_x[i]*pix_size, self.mapsize*pix_size-self.agents_y[i]*pix_size),
          ((self.agents_x[i]+self.agents_speed[i]/self.max_speed*np.cos(self.agents_heading[i]))*pix_size, self.mapsize*pix_size-(self.agents_y[i]+self.agents_speed[i]/self.max_speed*np.sin(self.agents_heading[i]))*pix_size),
          width=1,
        )
        # Acceleration vector
        pygame.draw.line(
          canvas,
          (0, 255, 0),
          (self.agents_x[i]*pix_size, self.mapsize*pix_size-self.agents_y[i]*pix_size),
          ((self.agents_x[i]+self.agents_accel[i]/self.max_accel*np.cos(self.agents_heading[i]+self.agents_angular_accel[i]))*pix_size, self.mapsize*pix_size-(self.agents_y[i]+self.agents_accel[i]/self.max_accel*np.sin(self.agents_heading[i]+self.agents_angular_accel[i]))*pix_size),
          width=1
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