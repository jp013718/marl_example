import numpy as np
import sys
import argparse
import gymnasium.spaces as spaces
from multiprocessing import Pool

sys.path.append("..")
from models.maddpg.maddpg import MADDPG
from models.maddpg.buffer import ReplayBuffer
from environment.envs.parallel_marl_env import ParallelMarlEnvironment

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


def get_states_from_infos(infos: dict):
  states = []
  for agent_idx in range(len(infos.values())):
    state = np.array(list(infos[f"agent_{agent_idx}"].values()))
    neighbors = list(infos.values())
    neighbors.pop(agent_idx)
    neighbors.sort(key=lambda dist: np.sqrt((dist['x']-infos[f'agent_{agent_idx}']['x'])**2+(dist['y']-infos[f'agent_{agent_idx}']['y'])**2))
    neighbors = np.array([np.array(list(neighbor.values())) for neighbor in neighbors])
    state = np.concat([state, *neighbors])
    states.append(state)

  return states
    

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-t', '--train', action='store_true')
  parser.add_argument('-c', '--checkpoint', default=None)
  parser.add_argument('-r', '--random_episodes', default=1000, type=int)
  parser.add_argument('-d', '--duration', default=50001, type=int)
  parser.add_argument('-n', '--num_agents', default=3, type=int)
  parser.add_argument('-k', '--k_near_agents', default=2, type=int)
  parser.add_argument('-m', '--minibatch_size', default=64, type=int)
  parser.add_argument('-e', '--eval', action='store_true')
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
  n_agents = args.num_agents
  agent_types = {0: "agent"}
  env = ParallelMarlEnvironment(n_agents=n_agents, num_near_agents=args.k_near_agents,render_mode=None)
  actor_dims = []
  n_actions = []
  for agent_type in agent_types.values():
    actor_dims.append(np.array(list(flatten_dict(env.observation_space(agent_type)).values())).shape[0])
    n_actions.append(env.action_space(agent_type).shape[0])
  critic_dims = len(list(flatten_dict(env._get_info(0))))*n_agents

  maddpg_agents = MADDPG(actor_dims[0], critic_dims, n_agents, 0, n_actions[0], minibatch_size=args.minibatch_size, fc1=args.fc1, fc2=args.fc2, alpha=args.alpha, beta=args.beta, gamma=args.gamma, tau=args.tau, scenario=scenario, chkpt_dir='tmp/maddpg/')

  memories = ReplayBuffer(1000000, critic_dims, actor_dims[0], n_actions[0], n_agents, batch_size=1024*n_agents)

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
    with Pool(n_agents) as p:
      reset_vals = p.map(env.reset, [agent_idx for agent_idx in range(n_agents)])

    # print(reset_vals)
    obs = {f'agent_{i}': val[0] for i, val in enumerate(reset_vals)}
    infos = {f'agent_{i}': val[1] for i, val in enumerate(reset_vals)}
    obs = unpack_dict(obs)
    score = 0
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    episode_step = 0
    done = [False]*n_agents 


    while not all(done):
      if evaluate:
        env.render()

      if i < N_EXPLORATION_GAMES and args.train and args.checkpoint is not None:
        # actions_dict = {}
        with Pool(n_agents) as p:
          actions = p.starmap(env.action_space('agent').sample, [() for _ in range(n_agents)])
      else:
        actions = maddpg_agents.choose_action(obs)
        actions = actions*np.array([env.max_angular_speed, env.max_speed/2])+np.array([0, env.max_speed/2])
      
      actions_list = []
      for agent_idx, term in enumerate(terminated.values()):
        if term:
          actions_list.append(np.array([0, -1]))
          actions[agent_idx] = np.array([0, -1])
        else:
          actions_list.append((actions[agent_idx]))

      if args.eval:
        print(actions)

      with Pool(n_agents) as p:
        step_result = p.starmap(env.step, [(agent_idx, actions_list[agent_idx]) for agent_idx in range(n_agents)])
        # obs_, rewards, terminated, truncated, infos_ = env.step(actions_dict)
      obs_ = {f'agent_{i}': val[0] for i, val in enumerate(step_result)}
      rewards = {f'agent_{i}': val[1] for i, val in enumerate(step_result)}
      terminated = {f'agent_{i}': val[2] for i, val in enumerate(step_result)}
      truncated = {f'agent_{i}': val[3] for i, val in enumerate(step_result)}
      infos_ = {f'agent_{i}': val[4] for i, val in enumerate(step_result)}
      obs_ = unpack_dict(obs_)

      states = get_states_from_infos(infos)
      states_ = get_states_from_infos(infos_)

      done = list(terminated.values()) if not all(truncated.values()) else list(truncated.values())
      rewards_list = np.array(list(rewards.values()))

      memories.store_transition(
        obs,
        states,
        actions,
        rewards_list,
        states_,
        done
      )

      if total_steps % 100 == 0 and not evaluate:
        maddpg_agents.learn(memories)

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