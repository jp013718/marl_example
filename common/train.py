import numpy as np
import sys
sys.path.append("..")
from models.maddpg.maddpg import MADDPG
from models.maddpg.utils import ReplayBuffer
from environment.envs.marl_env import MarlEnvironment

def obs_list_to_state_vector(observation):
  state = np.array([])
  for obs in observation:
    state = np.concatenate(
      [
        state,
        np.array(
          [
            observation[obs]["x"],
            observation[obs]["y"], 
            observation[obs]["heading"], 
            observation[obs]["speed"], 
            observation[obs]["target_heading"],
            observation[obs]["target_dist"]
          ]
        )
      ]
    )

  return state

if __name__ == "__main__":
  scenario = "simple"
  env = MarlEnvironment(render_mode=None)
  n_agents = env.num_agents
  actor_dims = []
  for i in range(n_agents):
    actor_dims.append(env.observation_space("agent"))
    print(dict(env.observation_space("agent")))
  critic_dims = n_agents*6

  n_actions = env.action_space('agent')
  maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, fc1=64, fc2=64, alpha=0.01, beta=0.01, scenario=scenario, chkpt_dir='tmp/maddpg/')

  memory = ReplayBuffer(1000000, critic_dims, actor_dims, n_actions, n_agents, batch_size=1024)

  PRINT_INTERVAL = 500
  N_GAMES = 50000
  total_steps = 0
  score_history = []
  evaluate = False
  best_score = 0

  if evaluate:
    maddpg_agents.load_checkpoint()

  for i in range(N_GAMES):
    obs, infos = env.reset()
    score = 0
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}
    episode_step = 0
    done = [False]*n_agents

    while not all(done):
      if evaluate:
        env.render()

      actions = maddpg_agents.choose_action(obs)
      obs_, rewards, terminated, truncated, infos = env.step(actions)
      state = obs_list_to_state_vector(obs)
      state_ = obs_list_to_state_vector(obs_)

      done = terminated.values() if not all(truncated.values()) else truncated.values()

      memory.store_transition(obs, state, actions, rewards, obs_, state_, done)

      if total_steps % 100 == 0 and not evaluate:
        maddpg_agents.learn(memory)

      obs = obs_

      score += sum(rewards)
      total_steps += 1
      episode_step += 1

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
    if not evaluate:
      if avg_score > best_score:
        maddpg_agents.save_checkpoint()
        best_score = avg_score
    if i % PRINT_INTERVAL == 0 and i > 0:
      print(f'episode: {i}; avg_score: {avg_score:.1f}')