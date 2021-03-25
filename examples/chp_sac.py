from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC
import logging
import gym
import numpy as np

path = "../../Dymola/IEEE14_014_0Buses_0CHP.fmu"
env_entry_point = 'examples:CHPEnv'

vref_2 = 1.0
vref_3 = 1.0
vref_6 = 1.0
vref_8 = 1.0
time_step = .5
positive_reward = 1
negative_reward = -100
log_level = logging.DEBUG

config = {
    'path': path,
    'vref_2': vref_2,
    'vref_3': vref_3,
    'vref_6': vref_6,
    'vref_8': vref_8,
    'time_step': time_step,
    'positive_reward': positive_reward,
    'negative_reward': negative_reward,
    'log_level': log_level
}

from gym.envs.registration import register
env_name = "CHPEnv-v0"

register(
    id=env_name,
    entry_point=env_entry_point,
    kwargs=config
)

env = gym.make(env_name)
model = SAC(MlpPolicy, env, verbose=1, tensorboard_log="tensorboard_logs")

min_reward = np.inf
max_reward = -np.inf
avg_reward = 0
obs = env.reset()
for _ in range(20):
    action = [0.0]
    my_action = np.multiply(0.05,action) + 1
    obs, reward, done, info = env.step(my_action)
    if done:
        env.reset()
    avg_reward += 1/30 * reward
    if reward < min_reward:
        min_reward = reward
    if reward > max_reward:
        max_reward = reward

env.max_reward = max_reward
env.min_reward = min_reward
env.avg_reward = avg_reward

model.learn(total_timesteps=1000, tb_log_name="chp")

obs = env.reset()
rl_reward = 0
for _ in range(20):
    action, _state = model.predict(obs)
    my_action = np.multiply(0.05,action) + 1
    obs, reward, done, info = env.step(my_action)
    if done:
        env.reset()
    rl_reward += reward

obs = env.reset()
base_reward = 0
for _ in range(20):
    # action, _state = model.predict(obs)
    action = [1.0]
    my_action = np.multiply(0.05,action) + 1
    obs, reward, done, info = env.step(my_action)
    if done:
        env.reset()
    base_reward += reward

print(f"{rl_reward},{base_reward}")
