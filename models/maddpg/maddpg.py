import torch
import torch.nn as nn
from agent import Agent

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, agent_type, n_actions, scenario='simple', alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        self.agent = Agent(
            actor_dims,
            critic_dims,
            n_actions,
            n_agents,
            agent_type,
            chkpt_dir,
            alpha,
            beta,
            fc1,
            fc2,
            gamma,
            tau
        )

    def save_checkpoint(self, location):
        print('...saving checkpoint...')
        self.agent.save_model(location)

    def load_checkpoint(self, location):
        print('...loading checkpoint...')
        self.agent.load_model(location)

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx in range(self.n_agents):
            actions.append(self.agent.choose_action(raw_obs[agent_idx]))

        return actions

    def learn(self):
        pass