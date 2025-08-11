import numpy as np
import gymnasium.spaces as spaces

class Agent:
  def __init__(
      self,
      name: str,
      x: np.float32=0.0, 
      y: np.float32=0.0,
      max_angular_accel: np.float32=0.2*np.pi,
      max_accel: np.float32=1,
      model_type: str|None=None,
      training: bool=False
  ):
    
    self.name = name
    self.x = x
    self.y = y
    self.max_angular_accel = max_angular_accel
    self.max_accel = max_accel
    self.speed = 0
    self.heading = 0
    self.angular_vel = 0
    self.accel = 0
    self.angular_accel = 0
    self.model_type = model_type
    self.training = training
    
    self.action_space = spaces.Box(low=np.array([-self.max_angular_accel, -self.max_accel]), high=np.array([self.max_angular_accel, self.max_accel]), shape=(2,), dtype=np.float64)

    if self.model_type:
      self._load_model(self.model_type)
    else:
      self.model = None

  def get_action(self, observation):
    if self.model:
      return self.model.get_action(observation)
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