import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

class Actor(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir):
        super(Actor, self).__init__()
        self.alpha = alpha
        self.chkpt_dir = chkpt_dir
        self.filename = name

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, x):
        x1 = nn.ReLU(self.fc1(x))
        x2 = nn.ReLU(self.fc2(x1))
        pi = nn.Tanh(self.pi(x2))

        return pi

    def save_checkpoint(self, location=''):
        Path(os.path.join(self.chkpt_dir, location)).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.chkpt_dir, location, self.filename))

    def load_checkpoint(self, location=''):
        self.load_state_dict(torch.load(os.path.join(self.chkpt_dir, location, self.filename)))

class Critic(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, chkpt_dir):
        self.beta = beta
        self.chkpt_dir = chkpt_dir
        self.filename = name

        self.fc1 = nn.Linear(input_dims+n_agents*n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optim = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, x):
        x1 = nn.ReLU(self.fc1(x))
        x2 = nn.ReLU(self.fc2(x1))
        q = self.q(x2)

        return q
    
    def save_checkpoint(self, location=''):
        Path(os.path.join(self.chkpt_dir, location)).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.chkpt_dir, location, self.filename))

    def load_checkpoint(self, location=''):
        self.load_state_dict(torch.load(os.path.join(self.chkpt_dir, location, self.filename)))
