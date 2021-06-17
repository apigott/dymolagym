from stable_baselines3.sac.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import SAC
import logging
import gym
import numpy as np
import os
import matplotlib.pyplot as plt
import DyMat
import time

# add reference libraries here. Current structure will use the relative path from this file
libs = ["../../../OpenIPSL-1.5.0/OpenIPSL/package.mo",
       "../../../OpenIPSL-1.5.0/ApplicationExamples/IEEE9/package.mo"]

# check that all the paths to library package.mo files exist
# DymolaInterface() also checks this but I've found this warning helpful
for lib in libs:
    if not os.path.isfile(lib):
        print(f"Cannot find the library {lib}")

mo_name = "IEEE9.IEEE_9_wVariation" # name of Modelica model in the Library.Model format
env_entry_point = 'examples:IEEE9Env' # Python package location of RL environment

time_step = 5 # time delta in seconds
positive_reward = 1
negative_reward = -100 # penalize RL agent for is_done
log_level = 0
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
    'default_action': default_action,
    'method':'Dassl'
}

# enable the model specific class as an OpenAI gym environment
from gym.envs.registration import register
env_name = "MicrogridEnv-v0"

register(
    id=env_name,
    entry_point=env_entry_point,
    kwargs=config
)

# create the environment. this will run an initial step and must return [True, [...]] or something is broken
# TODO: create error handling/warnings if simulations don't work (i.e. returns [False], [...])
env = gym.make(env_name)

env.reset()
env.reset_dymola()
for _ in range(30):
    action = np.random.uniform(-1,1,3).tolist()
    obs, reward, done, info = env.step(action)

mode = 'learn'

if mode == 'load':
    model = SAC.load("IEEE9_5k_v3", env=env)
else:
    model = SAC(MlpPolicy, env, learning_rate=10**-4, verbose=1, tensorboard_log='tensorboard_log')

tic = time.time()

env.reset()
for i in range(10):
    print(i)
    model.learn(1000, reset_num_timesteps=False)
    model.save("IEEE9_5k_v5")

toc = time.time()
print(toc-tic)

obs = env.reset()
for _ in range(50):
    action = model.predict(obs)[0]
    obs, reward, done, info = env.step(action)

legend = []
fig, ax = plt.subplots(4,3,figsize=(20,15))
for i in range(3):
    for j in range(3):
        bus = 1 + 3*i + j
        ax[i][j].plot(env.debug_data['my_time'], env.debug_data[f'B{bus}.V'], color='r')
        ax[i][j].set_xlabel('Time (sec)')
        ax[i][j].set_ylabel('Voltage (p.u.)')
        ax[i][j].set_title(f'Bus {bus}')
legend += ['RL Agent']
gen = ['G1.gENSAL.P','G2.gENROU.P','G3.gENROU.P']
for j in range(3):
    ax[3][j].plot(env.debug_data['my_time'], env.debug_data[gen[j]], color='r')
    ax[3][j].set_xlabel('Time (sec)')
    ax[3][j].set_ylabel('Generator Power Output (p.u.)')
    ax[3][j].set_title(gen[j])

env.reset()
for _ in range(50):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

for i in range(3):
    for j in range(3):
        bus = 1 + 3*i + j
        ax[i][j].plot(env.debug_data['my_time'], env.debug_data[f'B{bus}.V'], color='b')

legend += ['Randomized']
for j in range(3):
    ax[3][j].plot(env.debug_data['my_time'], env.debug_data[gen[j]], color='b')

env.reset()
for _ in range(50):
    action = [0,0,0] # null action
    obs, reward, done, info = env.step(action)

for i in range(3):
    for j in range(3):
        bus = 1 + 3*i + j
        ax[i][j].plot(env.debug_data['my_time'], env.debug_data[f'B{bus}.V'], color='g')

legend += ['Do Nothing']
for j in range(3):
    ax[3][j].plot(env.debug_data['my_time'], env.debug_data[gen[j]], color='g')

plt.savefig('test')
plt.show()
