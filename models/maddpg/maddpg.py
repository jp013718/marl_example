import torch
import torch.nn.functional as functional
from models.maddpg.agent import Agent
from models.maddpg.utils import ReplayBuffer


class MADDPG:
  def __init__(self, actor_dims, critic_dims, n_agents, n_actions, scenario='simple', alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01, chkpt_dir='/tmp/maddpg/'):
    self.n_agents = n_agents
    self.n_actions = n_actions
    chkpt_dir += scenario
    self.agents = [Agent(actor_dims[agent_idx], critic_dims, n_actions, n_agents, agent_idx, alpha=alpha, beta=beta, chkpt_dir=chkpt_dir) for agent_idx in range(self.n_agents)]

  
  def save_checkpoint(self, dir=''):
    print('...saving checkpoint...')
    for agent in self.agents:
      agent.save_models(dir)

  
  def load_checkpoint(self, dir=''):
    print('...loading checkpoint...')
    for agent in self.agents:
      agent.load_models(dir)

  
  def choose_action(self, raw_obs):
    actions = []
    for agent_idx, agent in enumerate(self.agents):
      action = agent.choose_action(raw_obs[agent_idx])
      actions.append(action)
    return actions
  

  def learn(self, memory: ReplayBuffer):
    if not memory.ready():
      return
    
    actor_states, states, actions, rewards, actor_new_states, states_, dones = memory.sample_buffer()

    device = self.agents[0].actor.device

    states = torch.tensor(states, dtype=torch.float).to(device)
    actions = torch.tensor(actions, dtype=torch.float).to(device)
    rewards = torch.tensor(rewards).to(device)
    states_ = torch.tensor(states_, dtype=torch.float).to(device)
    dones = torch.tensor(dones).to(device)

    all_agents_new_actions = []
    all_agents_new_mu_actions = []
    old_agents_actions = []

    for agent_idx, agent in enumerate(self.agents):
      new_states = torch.tensor(actor_new_states[agent_idx], dtype=torch.float).to(device)
      new_pi = agent.target_actor.forward(new_states)
      all_agents_new_actions.append(new_pi)
      mu_states = torch.tensor(actor_states[agent_idx], dtype=torch.float).to(device)
      pi = agent.actor.forward(mu_states)
      all_agents_new_mu_actions.append(pi)
      old_agents_actions.append(actions[agent_idx])

    new_actions = torch.cat([acts for acts in all_agents_new_actions], dim=1)
    mu = torch.cat([acts for acts in all_agents_new_mu_actions], dim=1)
    old_actions = torch.cat([acts for acts in old_agents_actions], dim=1)

    loss = torch.nn.MSELoss()

    for agent_idx, agent in enumerate(self.agents):
      critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
      # critic_value_[dones[:,0]] = 0.0
      critic_value = agent.critic.forward(states, old_actions).flatten().to(torch.double)

      target = rewards[:,agent_idx] + agent.gamma*critic_value_
      critic_loss = loss(target, critic_value)
      agent.critic.optimizer.zero_grad()
      critic_loss.backward(retain_graph=True)

      actor_loss = agent.critic.forward(states, mu).flatten()
      actor_loss = -torch.mean(actor_loss)
      agent.actor.optimizer.zero_grad()
      actor_loss.backward(retain_graph=True)

    
    for agent in self.agents:
      agent.critic.optimizer.step()
      agent.actor.optimizer.step()
      agent.update_network_parameters()