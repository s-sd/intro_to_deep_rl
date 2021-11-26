import gym

import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

# =============================================================================
# solving Cartpole
# =============================================================================

env = gym.make('CartPole-v0')

# test cartpole with a random policy
initial_obs = env.reset()
obs_shape = np.shape(initial_obs)
for _ in range(500):
    env.render()
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    if done:
        env.reset()
env.close()

# prepare the envrionment and model for trianing
env = make_vec_env('CartPole-v1', n_envs=1)
model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)

# testing the trained model
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
