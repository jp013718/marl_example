import numpy as np
import sys
import gymnasium.spaces as spaces
sys.path.append("..")
from models.maddpg.maddpg import MADDPG
from models.maddpg.utils import ReplayBuffer
from environment.envs.marl_env import MarlEnvironment

import torch
torch.autograd.set_detect_anomaly(True)

def obs_list_to_state_vector(observation):
  state = np.array([])
  for obs in observation:
    state = np.concatenate([state, obs])
  return state


def flatten_dict(d: dict, prev_key=''):
  new_dict = {}
  for key, val in d.items():
    if type(val) is dict or type(val) is spaces.Dict:
      new_dict.update(flatten_dict(val, (prev_key+'_'+key) if prev_key != '' else key))
    else:
      new_dict.update({(prev_key+'_'+key) if prev_key != '' else key: val})
  return new_dict


def unpack_obs_dict(obs: dict):
  obs_list = []
  for val in obs.values():
    indv_obs_dict = flatten_dict(val)
    obs_list.append(list(indv_obs_dict.values()))

  return obs_list


def action_list_to_action_dict(actions):
  action_dict = {}
  for i, action in enumerate(actions):
    action_dict.update({f'agent_{i}': action})
  return action_dict


if __name__ == "__main__":
  scenario = "simple"
  env = MarlEnvironment(render_mode=None)
  n_agents = env.num_agents
  actor_dims = []
  for i in range(n_agents):
    actor_dims.append(np.array(list(flatten_dict(env.observation_space("agent")).values())).shape[0])
  critic_dims = sum(actor_dims)

  n_actions = env.action_space('agent').shape[0]
  maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, fc1=64, fc2=64, alpha=0.01, beta=0.01, scenario=scenario, chkpt_dir='tmp/maddpg/')

  memory = ReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size=1024)

  PRINT_INTERVAL = 10
  N_GAMES = 50000
  total_steps = 0
  score_history = []
  evaluate = True
  eval_model = "best"
  save_freq = 500
  best_score = -np.inf

  if evaluate:
    maddpg_agents.load_checkpoint("recent")
    env.render_mode = "human"

  for i in range(N_GAMES):
    obs, infos = env.reset()
    obs = unpack_obs_dict(obs)
    score = 0
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    episode_step = 0
    done = [False]*n_agents

    while not all(done):
      if evaluate:
        env.render()

      actions = maddpg_agents.choose_action(obs)
      actions_dict = action_list_to_action_dict(actions)
      obs_, rewards, terminated, truncated, infos = env.step(actions_dict)
      obs_ = unpack_obs_dict(obs_)
      state = obs_list_to_state_vector(obs)
      state_ = obs_list_to_state_vector(obs_)

      done = terminated.values() if not all(truncated.values()) else truncated.values()
      rewards_list = np.array(list(rewards.values()))

      memory.store_transition(obs, state, actions, rewards_list, obs_, state_, done)

      if total_steps % 100 == 0 and not evaluate:
        maddpg_agents.learn(memory)

      obs = obs_

      score += sum(rewards_list)
      total_steps += 1
      episode_step += 1

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    if not evaluate:
      if avg_score > best_score and len(score_history) >= 100:
        maddpg_agents.save_checkpoint("best")
        best_score = avg_score
      if i % save_freq == 0 and len(score_history) > 0: 
        maddpg_agents.save_checkpoint("recent")
    if i % PRINT_INTERVAL == 0 and i > 0:
      print(f'episode: {i}; avg_score: {avg_score:.1f}')