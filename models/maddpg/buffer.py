import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.memory = np.array([{}]*max_size)

    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        mem_idx = self.mem_cntr % self.mem_size
        self.memory[mem_idx].update(
            {
                "observation": raw_obs,
                "state": state,
                "action": action,
                "reward": reward,
                "observation_": raw_obs_,
                "state_": state_,
                "done": done,
            }
        )
            
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        return self.memory[batch]


    def ready(self):
        return self.mem_cntr >= self.batch_size

