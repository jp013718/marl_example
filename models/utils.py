import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pettingzoo import ParallelEnv
import numpy as np

def flatten_dict_to_list(d: dict):
  lst = []
  for item in d.values():
    if type(item) is dict:
      contents = flatten_dict_to_list(item)
      for val in contents:
        lst.append(val)
    else:
      lst.append(item)

  return lst

def flatten_dict_without_nest(d: dict):
  lst = []
  for item in d.values():
    if not type(item) is dict:
      lst.append(item)

  return lst

class Model:
  def __init__(self):
    raise NotImplementedError()

  def get_action(self, observation):
    return NotImplementedError()


class Actor(nn.Module):
  def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir):
    super(Actor, self).__init__()

    self.chkpt_file = os.path.join(chkpt_dir, name)

    self.fc1 = nn.Linear(input_dims, fc1_dims)
    self.fc2 = nn.Linear(fc1_dims, fc2_dims)
    self.pi = nn.Linear(fc1_dims, n_actions)

    self.optimizer = optim.Adam(self.parameters, lr=alpha)
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    self.to(self.device)

  def forward(self, state):
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    pi = torch.softmax(self.pi(x), dim=1)

    return pi
  
  def save_checkpoint(self):
    torch.save(self.state_dict(), self.chkpt_file)

  def load_checkpoint(self):
    self.load_state_dict(torch.load(self.chkpt_file))


class Critic(nn.Module):
  def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, chkpt_dir):
    super(Critic, self).__init__()

    self.chkpt_file = os.path.join(chkpt_dir, name)

    self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
    self.fc2 = nn.Linear(fc1_dims, fc2_dims)
    self.q = nn.Linear(fc2_dims, 1)

    self.optimizer = optim.Adam(self.parameters(), lr=beta)
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    self.to(self.device)

  def forward(self, state, action):
    x = F.relu(self.fc1(torch.cat([state, action], dim=1)))
    x = F.relu(self.fc2(x))
    q = self.q(x)

    return q
  
  def save_checkpoint(self):
    torch.save(self.state_dict(), self.chkpt_file)

  def load_checkpoint(self):
    self.load_state_dict(torch.load(self.chkpt_file))