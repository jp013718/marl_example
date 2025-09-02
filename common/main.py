import numpy as np
import sys
import argparse
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


def unpack_dict(obs: dict):
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
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--train', action='store_true')
  parser.add_argument('-c', '--checkpoint', default=None)
  parser.add_argument('-r', '--random_episodes', default=1000, type=int)
  parser.add_argument('-d', '--duration', default=50001, type=int)
  parser.add_argument('-n', '--num_agents', default=3, type=int)
  parser.add_argument('-k', '--k_near_agents', default=2, type=int)
  parser.add_argument('--fc1', default=64, type=int)
  parser.add_argument('--fc2', default=64, type=int)
  parser.add_argument('--alpha', default=0.01, type=float)
  parser.add_argument('--beta', default=0.01, type=float)
  parser.add_argument('--gamma', default=0.99, type=float)
  parser.add_argument('--tau', default=0.01, type=float)

  args = parser.parse_args()
  
  if not args.train:
    try:
      assert args.checkpoint
    except AssertionError as e:
      raise e("No checkpoint specified for evaluation. Use the flag -c [checkpoint] or --checkpoint [checkpoint]")

  scenario = "simple"
  n_agents = [int(args.num_agents)]
  agent_types = {0: "agent"}
  env = MarlEnvironment(n_agents=n_agents[0], num_near_agents=args.k_near_agents,render_mode=None)
  actor_dims = []
  n_actions = []
  for agent_type in agent_types.values():
    actor_dims.append(np.array(list(flatten_dict(env.observation_space(agent_type)).values())).shape[0])
    n_actions.append(env.action_space(agent_type).shape[0])
  critic_dims = len(list(flatten_dict(env._get_infos())))

  maddpg_agents = MADDPG(actor_dims, critic_dims, 1, n_agents, n_actions, fc1=args.fc1, fc2=args.fc2, alpha=args.alpha, beta=args.beta, gamma=args.gamma, tau=args.tau, scenario=scenario, chkpt_dir='tmp/maddpg/')

  memories = [ReplayBuffer(1000000, critic_dims, actor_dims[agent_type], n_actions[agent_type], n_agents[agent_type], batch_size=1024*n_agents[i]) for i, agent_type in enumerate(agent_types.keys())]

  PRINT_INTERVAL = 10
  N_GAMES = args.duration
  N_EXPLORATION_GAMES = args.random_episodes
  total_steps = 0
  score_history = []
  evaluate = not args.train
  load_model = not (args.checkpoint is None)
  eval_model = args.checkpoint
  save_freq = 500
  best_score = -np.inf

  if load_model:
    maddpg_agents.load_checkpoint(eval_model)
  
  if evaluate:  
    env.render_mode = "human"

  for i in range(N_GAMES):
    obs, infos = env.reset()
    obs = unpack_dict(obs)
    infos = flatten_dict(infos)
    score = 0
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    episode_step = 0
    done = [False]*sum(n_agents)

    while not all(done):
      if evaluate:
        env.render()

      if i < N_EXPLORATION_GAMES and args.train:
        actions_dict = {}
        for agent in env.agents:
          actions_dict[agent] = env.action_space("agent").sample() if not (terminated[agent] or truncated[agent]) else np.array([0,0], dtype=np.float64)
        actions = np.array(list(actions_dict.values()), dtype=np.float64)/np.array([env.max_angular_accel, env.max_accel], dtype=np.float64)
      else:
        actions = maddpg_agents.choose_action(obs)
        actions_list = []
        for agent_idx, term in enumerate(terminated.values()):
          if term:
            actions_list.append(np.array([0, 0]))
            actions[agent_idx] = np.array([0, 0])
          else:
            actions_list.append(actions[agent_idx]*np.array([env.max_angular_accel, env.max_accel]))
        actions_dict = action_list_to_action_dict(actions_list)

      # print(actions)

      obs_, rewards, terminated, truncated, infos_ = env.step(actions_dict)
      obs_ = unpack_dict(obs_)
      infos_ = flatten_dict(infos_)
      state = np.array(list(infos.values()))
      state_ = np.array(list(infos_.values()))
      # state = obs_list_to_state_vector(obs)
      # state_ = obs_list_to_state_vector(obs_)

      done = list(terminated.values()) if not all(truncated.values()) else list(truncated.values())
      rewards_list = np.array(list(rewards.values()))

      for agent_type, n_agent_type in enumerate(n_agents):
        slice_start = sum(n_agents[0:agent_type])
        slice_end = sum(n_agents[0:agent_type])+n_agent_type+1
        memories[agent_type].store_transition(
          obs[slice_start:slice_end], 
          state, 
          actions[slice_start:slice_end], 
          rewards_list[slice_start:slice_end], 
          obs_[slice_start:slice_end], 
          state_, 
          done
        )

        if total_steps % 100 == 0 and not evaluate:
          maddpg_agents.learn(agent_type, memories[agent_type])

      obs = obs_
      infos = infos_

      score += sum(rewards_list)
      total_steps += 1
      episode_step += 1

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    if not evaluate:
      if avg_score > best_score and i >= 100 and i > N_EXPLORATION_GAMES:
        maddpg_agents.save_checkpoint("best")
        best_score = avg_score
      if i % save_freq == 0 and i > 1: 
        maddpg_agents.save_checkpoint("recent")
    if i % PRINT_INTERVAL == 0 and i > 0:
      print(f'episode: {i}; avg_score: {avg_score:.1f}')