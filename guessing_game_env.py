import gym

import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import A2C

# =============================================================================
# Creating and solving a text-based guessing game envrionment
# The environment in this script is adapted from: 
# https://github.com/JKCooper2/gym-envs/blob/master/GuessingGame/guessing_game.py
# =============================================================================

# We have a random number and we want to guess it within 200 tries
# The observation is guess being: higher [1], equal [2], lower [3]; than the number
# If we successfully guess a number that is very close to the number then we get +1 reward
# The action is a guessed number


class SimpleEnv2(gym.Env):
    def __init__(self):
        self.range = 1000  # Randomly selected number is within +/- this value
        self.bounds = 10000

        self.action_space = gym.spaces.Box(low=np.array([-self.bounds]), high=np.array([self.bounds]))
        self.observation_space = gym.spaces.Discrete(4)

        self.number = 0
        self.guess_count = 0
        self.guess_max = 200
        self.observation = 0

    def step(self, action):
        assert self.action_space.contains(action)
        if action < self.number:
            self.observation = 1
        elif action == self.number:
            self.observation = 2
        elif action > self.number:
            self.observation = 3
        reward = 0
        done = False

        if (self.number - self.range * 0.01) < action < (self.number + self.range * 0.01):
            reward = 1
            done = True

        self.guess_count += 1
        if self.guess_count >= self.guess_max:
            done = True

        return self.observation, reward, done, {}

    def reset(self):
        self.number = np.random.uniform(-self.range, self.range)
        self.guess_count = 0
        self.observation = 0
        return self.observation


# a function to test how well the agent does
def eval_agent(eval_steps):
    obs = env.reset()
    guess_count_list = []
    for _ in range(eval_steps):
        number = env.number
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            guess_count = env.guess_count
            guess_count_list.append(guess_count)
            env.reset()
            # print(f'The agent guessed the number {number} in {guess_count} tries')
    print(f'The average guess count was {np.mean(guess_count_list)}')
    return np.mean(guess_count_list)


env = SimpleEnv2()

# testing a random policy
initial_obs = env.reset()
obs_shape = np.shape(initial_obs)
for _ in range(4):
    # env.render()
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print(f'The observation is {obs} and the agent picked {action}')
    if done:
        env.reset()

# preparing the model
model = A2C(MlpPolicy, env, verbose=0)

# combined learning and evaluation as 'trials'
num_trials = 20
for trial in range(num_trials):
    model.learn(total_timesteps=4000)
    print(f'Trial {trial}')
    avg_guess_count = eval_agent(1000)
    

