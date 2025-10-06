import sys
sys.path.append("..")

from environment.envs.marl_env import MarlEnvironment

if __name__ == "__main__":
  env = MarlEnvironment(
    mapsize=50
  )

  episodes = 10
  completed_episodes = 0

  while completed_episodes < episodes:
    observations, infos = env.reset()
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}

    while not (all(terminated.values()) or all(truncated.values())):
      env.render()

      actions = {}
      for agent in env.agents:
        actions[agent] = env.action_space("agent").sample() if not (terminated[agent] or truncated[agent]) else None

      observations, rewards, terminated, truncated, infos = env.step(actions)
    
    completed_episodes += 1

  env.close()

