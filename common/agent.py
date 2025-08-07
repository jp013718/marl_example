import numpy as np
import gymnasium.spaces as spaces

class Agent:
  def __init__(
      self,
      name: str,
      action_space: spaces.Space,
      x: np.float32=0.0, 
      y: np.float32=0.0,
      model_type: str|None=None
  ):
    
    self.name = name
    self.action_space = action_space
    self.x = x
    self.y = y
    self.speed = 0
    self.heading = 0
    self.angular_vel = 0
    self.accel = 0
    self.angular_accel = 0
    self.model_type = model_type
    
    if self.model_type:
      self._load_model(self.model_type)
    else:
      self.model = None

  def get_action(self, observation):
    if self.model:
      self.model.get_action(observation)
    else:
      return self.action_space.sample()

  def reset(self):
    self.x = 0
    self.y = 0
    self.speed = 0
    self.accel = 0
    self.heading = 0
    self.angular_vel = 0
    self.angular_accel = 0

  def _load_model(self, model):
    raise NotImplementedError("Model selection not implemented")