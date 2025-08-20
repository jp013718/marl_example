import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path

class Actor(nn.Module):
  def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir):
    super(Actor, self).__init__()

    
    self.chkpt_dir = chkpt_dir
    self.chkpt_file = name

    self.fc1 = nn.Linear(input_dims, fc1_dims)
    self.fc2 = nn.Linear(fc1_dims, fc2_dims)
    self.pi = nn.Linear(fc1_dims, n_actions)

    self.optimizer = optim.Adam(self.parameters(), lr=alpha)
    # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.device = 'cpu'

    self.to(self.device)

  def forward(self, state):
    x1 = F.relu(self.fc1(state))
    x2 = F.relu(self.fc2(x1))
    pi = torch.softmax(self.pi(x2), dim=1)

    return pi
  
  def save_checkpoint(self, dir=''):
    Path(os.path.join(self.chkpt_dir, dir)).mkdir(parents=True, exist_ok=True)
    torch.save(self.state_dict(), os.path.join(self.chkpt_dir, dir, self.chkpt_file))

  def load_checkpoint(self, dir=''):
    self.load_state_dict(torch.load(os.path.join(self.chkpt_dir, dir, self.chkpt_file)))


class Critic(nn.Module):
  def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, chkpt_dir):
    super(Critic, self).__init__()

    self.chkpt_dir = chkpt_dir
    self.chkpt_file = name

    self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
    self.fc2 = nn.Linear(fc1_dims, fc2_dims)
    self.q = nn.Linear(fc2_dims, 1)

    self.optimizer = optim.Adam(self.parameters(), lr=beta)
    # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.device = 'cpu'

    self.to(self.device)

  def forward(self, state, action):
    x1 = F.relu(self.fc1(torch.cat([state, action], dim=1)))
    x2 = F.relu(self.fc2(x1))
    q = self.q(x2)

    return q
  
  def save_checkpoint(self, dir=''):
    Path(os.path.join(self.chkpt_dir, dir)).mkdir(parents=True, exist_ok=True)
    torch.save(self.state_dict(), os.path.join(self.chkpt_dir, dir, self.chkpt_file))

  def load_checkpoint(self, dir=''):
    self.load_state_dict(torch.load(os.path.join(self.chkpt_dir, dir, self.chkpt_file)))


class ReplayBuffer:
  def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size):
    self.mem_size = max_size
    self.mem_cntr = 0
    self.n_agents = n_agents
    self.actor_dims = actor_dims
    self.batch_size = batch_size
    self.n_actions = n_actions

    self.state_memory = np.zeros((self.mem_size, critic_dims))
    self.new_state_memory = np.zeros((self.mem_size, critic_dims))
    self.reward_memory = np.zeros((self.mem_size, n_agents))
    self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

    self.init_actor_memory()

  def init_actor_memory(self):
    self.actor_state_memory = []
    self.actor_new_state_memory = []
    self.actor_action_memory = []

    for i in range(self.n_agents):
      self.actor_state_memory.append(np.zeros((self.mem_size, self.actor_dims)))
      self.actor_new_state_memory.append(np.zeros((self.mem_size, self.actor_dims)))
      self.actor_action_memory.append(np.zeros((self.mem_size, self.n_actions)))

  def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
    index = self.mem_cntr % self.mem_size
    for agent_idx in range(self.n_agents):
      self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
      self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
      self.actor_action_memory[agent_idx][index] = action[agent_idx]

    self.state_memory[index] = state
    self.new_state_memory[index] = state_
    self.reward_memory[index] = reward
    self.terminal_memory[index] = done
    self.mem_cntr += 1

  def sample_buffer(self):
    max_mem = min(self.mem_cntr, self.mem_size)

    batch = np.random.choice(max_mem, self.batch_size, replace=False)

    states = self.state_memory[batch]
    rewards = self.reward_memory[batch]
    states_ = self.new_state_memory[batch]
    terminal = self.terminal_memory[batch]

    actor_states = []
    actor_new_states = []
    actions = []
    for agent_idx in range(self.n_agents):
      actor_states.append(self.actor_state_memory[agent_idx][batch])
      actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
      actions.append(self.actor_action_memory[agent_idx][batch])

    return actor_states, states, actions, rewards, actor_new_states, states_, terminal
  
  def ready(self):
    return self.mem_cntr >= self.batch_size
  