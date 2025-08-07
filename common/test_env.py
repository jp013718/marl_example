import sys
sys.path.append("..")

from agent import Agent
from environment.envs.marl_env import MarlEnvironment

if __name__ == "__main__":
  agents = [Agent(name=f"agent_{i}") for i in range(5)]
  env = MarlEnvironment(
    agents_list=agents,
    mapsize=50
  )

  episodes = 10
  completed_episodes = 0

  while completed_episodes < episodes:
    observations, infos = env.reset()
    terminated = {agent.name: False for agent in agents}
    truncated = {agent.name: False for agent in agents}

    while not (all(terminated.values()) or all(truncated.values())):
      env.render()

      actions = {}
      for agent in agents:
        actions[agent.name] = agent.get_action(observations[agent.name]) if not (terminated[agent.name] or truncated[agent.name]) else None

      observations, rewards, terminated, truncated, infos = env.step(actions)
    
    completed_episodes += 1

  env.close()

