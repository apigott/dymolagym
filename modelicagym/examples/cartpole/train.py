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

def eval_model(env, model_name):
    env.reset()
    env.reset_dymola()

    mode = 'load'

    if mode == 'load':
        model = SAC.load(model_name, env=env)
    else:
        model = SAC(MlpPolicy, env, learning_rate=10**-4, verbose=1, tensorboard_log='tensorboard_log')
        tic = time.time()

        env.reset()
        model.learn(10000, reset_num_timesteps=False)
        model.save("IEEE9_5k_v4")

        toc = time.time()
        print(toc-tic)

    obs = env.reset()
    actions = []
    rewards = []
    for _ in range(250):
        action = model.predict(obs)[0]
        actions += action.tolist()
        obs, reward, done, info = env.step(action)
        rewards += [reward]
    volt_norm = []
    legend = []
    fig, ax = plt.subplots(5,3,figsize=(40,30))
    for i in range(3):
        for j in range(3):
            bus = 1 + 3*i + j
            bus_volt = np.array(env.debug_data[f'iEEE_14_Buses.B{bus}.V'])-1.0
            volt_norm += [bus_volt]
            ax[i][j].plot(env.debug_data['iEEE_14_Buses.my_time'], bus_volt, color='r')
            ax[i][j].set_ylabel('Voltage Dev')
            ax[i][j].set_title(f'Bus {bus}')

    legend += ['RL Agent']
    gen = ['G2.gENROU.P','G1.gENSAL.P']
    for j in range(2):
        ax[3][j].plot(env.debug_data['iEEE_14_Buses.my_time'], env.debug_data[f'iEEE_14_Buses.{gen[j]}'], color='r')
        ax[3][j].set_xlabel('Time (sec)')
        ax[3][j].set_title(f'{gen[j]} Output')
        ax[4][j].plot(np.arange(250), actions[j::2], color='r')
    ax[3][2].plot(env.debug_data['iEEE_14_Buses.my_time'],np.divide(np.cumsum(np.linalg.norm(volt_norm, axis=0)),np.clip(env.debug_data['iEEE_14_Buses.my_time'], 1, np.inf)), color='r')
    ax[3][2].plot(env.debug_data['iEEE_14_Buses.my_time'],np.linalg.norm(volt_norm, axis=0), color='r')
    ax[4][2].plot(np.arange(250), rewards, color='r')

    env.reset()
    actions = []
    rewards = []
    for _ in range(250):
        action = env.action_space.sample()
        actions += action.tolist()
        obs, reward, done, info = env.step(action)
        rewards += [reward]
    volt_norm = []
    for i in range(3):
        for j in range(3):
            bus = 1 + 3*i + j
            bus_volt = np.array(env.debug_data[f'iEEE_14_Buses.B{bus}.V'])-1.0
            volt_norm += [bus_volt]
            ax[i][j].plot(env.debug_data['iEEE_14_Buses.my_time'], bus_volt, color='b')
    legend += ['Randomized']
    for j in range(2):
        ax[3][j].plot(env.debug_data['iEEE_14_Buses.my_time'], env.debug_data[f'iEEE_14_Buses.{gen[j]}'], color='b')
        ax[4][j].plot(np.arange(250), actions[j::2], color='b')
    ax[3][2].plot(env.debug_data['iEEE_14_Buses.my_time'],np.linalg.norm(volt_norm, axis=0), color='b')
    ax[3][2].plot(env.debug_data['iEEE_14_Buses.my_time'],np.divide(np.cumsum(np.linalg.norm(volt_norm, axis=0)),np.clip(env.debug_data['iEEE_14_Buses.my_time'], 1, np.inf)), color='b')
    ax[4][2].plot(np.arange(250), rewards, color='b')

    env.reset()
    actions = []
    rewards = []
    for _ in range(250):
        action = env.default_action # null action
        actions += action
        obs, reward, done, info = env.step(action)
        rewards += [reward]

    volt_norm = []
    for i in range(3):
        for j in range(3):
            bus = 1 + 3*i + j
            bus_volt = np.array(env.debug_data[f'iEEE_14_Buses.B{bus}.V'])-1.0
            volt_norm += [bus_volt]
            ax[i][j].plot(env.debug_data['iEEE_14_Buses.my_time'], bus_volt, color='g')
    legend += ['Do Nothing']
    for j in range(2):
        ax[3][j].plot(env.debug_data['iEEE_14_Buses.my_time'], env.debug_data[f'iEEE_14_Buses.{gen[j]}'], color='g')
        ax[4][j].plot(np.arange(250), actions[j::2], color='g')
    ax[3][2].plot(env.debug_data['iEEE_14_Buses.my_time'],np.divide(np.cumsum(np.linalg.norm(volt_norm, axis=0)),np.clip(env.debug_data['iEEE_14_Buses.my_time'], 1, np.inf)), color='g')
    ax[3][2].plot(env.debug_data['iEEE_14_Buses.my_time'],np.linalg.norm(volt_norm, axis=0), color='g')
    ax[0][2].legend(legend)
    ax[4][2].plot(np.arange(250), rewards, color='g')

    env.dymola.close()

    plt.savefig(model_name)
    plt.show()
    return

def train_model(env, model_name):
    model = SAC(MlpPolicy, env, learning_rate=10**-5, ent_coef='auto_0.1', verbose=1, tensorboard_log='tensorboard_log', batch_size=64)

    env.reset()
    for _ in range(10):
        model.learn(1000, reset_num_timesteps=False)
        model.save(model_name)
    return

if __name__=='__main__':
    # these config values are passed to the model specific environment class
    # mo_name and libs are passed on to the DymolaBaseEnv class
    

    # enable the model specific class as an OpenAI gym environment


    # create the environment. this will run an initial step and must return [True, [...]] or something is broken
    # TODO: create error handling/warnings if simulations don't work (i.e. returns [False], [...])
    env = gym.make(env_name)

    train_model(env, model_name)
    eval_model(env, model_name)
