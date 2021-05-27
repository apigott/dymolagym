from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC
import logging
import gym
import numpy as np
import os
import time
from gym.envs.registration import register

# add reference libraries here. Current structure will use the relative path from this file
libs = ["../../OpenIPSL-1.5.0/OpenIPSL/package.mo"]

# check that all the paths to library package.mo files exist
# DymolaInterface() also checks this but I've found this warning helpful
for lib in libs:
    if not os.path.isfile(lib):
        print(f"Cannot find the library {lib}")

mo_name = "OpenIPSL.Examples.IEEE9.IEEE_9_Base_Case_OL" # name of Modelica model in the Library.Model format
env_entry_point = 'examples:IEEE9Env' # Python package location of RL environment

time_step = 1 # time delta in seconds
positive_reward = 1
negative_reward = -100 # penalize RL agent for is_done
log_level = logging.DEBUG
default_action = [0,0,0]

# these config values are passed to the model specific environment class
# mo_name and libs are passed on to the DymolaBaseEnv class
config = {
    'mo_name': mo_name,
    'libs': libs,
#     'actions': actions,
#     'states': states,
    'time_step': time_step,
    'positive_reward': positive_reward,
    'negative_reward': negative_reward,
    'log_level': log_level,
    'default_action': default_action
}

# enable the model specific class as an OpenAI gym environment
env_name = "MicrogridEnv-v0"

register(
    id=env_name,
    entry_point=env_entry_point,
    kwargs=config
)

# create the environment. this will run an initial step and must return [True, [...]] or something is broken
# TODO: create error handling/warnings if simulations don't work (i.e. returns [False], [...])
env = gym.make(env_name)

mode = 'learn'

if mode == 'load':
    model = SAC.load("IEEE9_5k_v2", env=env)
else:
    model = SAC(MlpPolicy, env, learning_rate=10**-6, learning_starts=1024, batch_size=512, verbose=1)

tic = time.time()

# for i in range(10):
model.env.reset()
model.learn(10000, reset_num_timesteps=False)
model.save("IEEE9_5k_v3")

toc = time.time()
print(toc-tic)
