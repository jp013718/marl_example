import torch
import torch.nn as nn
import numpy as np
from models.maddpg.buffer import ReplayBuffer
from models.maddpg.agent import Agent

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, agent_type, n_actions, minibatch_size=64, scenario='simple', alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01, chkpt_dir='tmp/maddpg/'):
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        self.minibatch_size = minibatch_size
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

    def learn(self, memory: ReplayBuffer):
        if not memory.ready():
            return
        
        memories = memory.sample_buffer()
        minibatches = np.array_split(memories, len(memories)//self.minibatch_size)

        device = self.agent.actor.device

        for minibatch in minibatches:
            observations = torch.tensor(np.array([entry['observation'] for entry in minibatch], dtype=np.float64)).type(torch.float).to(device)
            states = torch.tensor(np.array([entry['state'] for entry in minibatch], dtype=np.float64)).type(torch.float).to(device)
            actions = torch.tensor(np.array([entry['action'] for entry in minibatch], dtype=np.float64)).type(torch.float).to(device)
            rewards = torch.tensor(np.array([entry['reward'] for entry in minibatch], dtype=np.float64)).type(torch.float).to(device)
            observations_ = torch.tensor(np.array([entry['observation_'] for entry in minibatch], dtype=np.float64)).type(torch.float).to(device)
            states_ = torch.tensor(np.array([entry['state_'] for entry in minibatch], dtype=np.float64)).type(torch.float).to(device)
            dones = torch.tensor(np.array([entry['done'] for entry in minibatch], dtype=bool)).to(device)

            new_actions = []
            mu = []
            for i in range(self.n_agents):
              obs = observations[:,i,:]
              new_agent_actions = self.agent.target_actor.forward(obs).detach().cpu().numpy()
              new_mu_actions = self.agent.actor.forward(obs).detach().cpu().numpy()
              new_actions.append(new_agent_actions)
              mu.append(new_mu_actions)

            new_actions = np.array(new_actions)
            new_actions = torch.tensor(np.array([new_actions[:,i,:] for i in range(self.minibatch_size)])).to(device)
            mu = np.array(mu)
            mu = torch.tensor(np.array([mu[:,i,:] for i in range(self.minibatch_size)])).to(device)

            self.agent.critic.optimizer.zero_grad()
            self.agent.actor.optimizer.zero_grad()

            for agent_idx in range(self.n_agents):
              critic_input_ = torch.cat([states_, new_actions.reshape(self.minibatch_size, self.n_actions*self.n_agents)], dim=1)
              # print(critic_input_)
              critic_value_ = self.agent.target_critic.forward(critic_input_).flatten()
              # print(f"Target Critic Value: {critic_value_}")
              critic_value_[dones[:,agent_idx]] = 0.0
              critic_input = torch.cat([states, actions.reshape(self.minibatch_size, self.n_actions*self.n_agents)], dim=1)
              critic_value = self.agent.critic.forward(critic_input).flatten()
              # print(f"Critic Value: {critic_value}")

              target = rewards[:,agent_idx] + self.agent.gamma*critic_value_
              # print(f"Target: {target}")
              critic_loss = nn.functional.mse_loss(target, critic_value)
              # print(f"Critic Loss: {critic_loss}")
              critic_loss.backward(retain_graph=True)

              mu_critic_input = torch.cat([states, mu.reshape(self.minibatch_size, self.n_actions*self.n_agents)], dim=1)
              actor_loss = self.agent.critic.forward(mu_critic_input).flatten()
              actor_loss = -torch.mean(actor_loss)
              # print(f"Actor Loss: {actor_loss}")
              actor_loss.backward(retain_graph=True)
              
            self.agent.critic.optimizer.step()
            self.agent.actor.optimizer.step()
            self.agent.update_network_parameters()