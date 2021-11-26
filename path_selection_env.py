import gym

import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

# =============================================================================
# Creating and solving a text-based path selection game environment
# =============================================================================

# The observation is the cost of travelling on 4 paths
# The reward is 1 if the lowest costing path is chosen and 0 otherwise
# The action is to pick a path

class SimpleEnv(gym.Env):
    def __init__(self):
        self.num_paths = 4
        self.observation_space = gym.spaces.Box(shape=(self.num_paths,), high=1.0, low=0.0)
        self.action_space = gym.spaces.Discrete(self.num_paths)
        self.reward_map = np.zeros(self.num_paths)
    def update_reward_map(self, obs):
        self.reward_map[:] = 0
        self.reward_map[np.argmin(obs)] = 1
    def reset(self):
        obs = np.random.rand(self.num_paths)
        self.update_reward_map(obs)
        return obs
    def step(self, a):
        a = int(a)
        reward = self.reward_map[a]
        obs = np.random.rand(self.num_paths)
        self.update_reward_map(obs)
        return obs, reward, False, {}

# a function to test how well the agent does
def eval_agent(num_steps): 
    obs = env.reset()
    total_count = 0
    correct_count = 0
    for _ in range(num_steps):
        action, _states = model.predict(obs)
        if int(np.argmin(obs)) == int(action):
            correct_count += 1
        obs, rewards, dones, info = env.step(action)
        total_count += 1
    print(f'Ratio of samples correctly identified: {correct_count/total_count}')

    
env = SimpleEnv()

# testing a random policy
initial_obs = env.reset()
obs_shape = np.shape(initial_obs)
for _ in range(4):
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print(f'The observation is {obs} and the agent picked {action}')
    if done:
        env.reset()
        
# preparing the model
model = A2C(MlpPolicy, env, verbose=0)

# combined learning and evaluation as 'trials'
num_trials = 10
for trial in range(num_trials):
    model.learn(total_timesteps=2000)
    print(f'Trial: {trial}')
    eval_agent(1000)
    
