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

    model = SAC.load(model_name, env=env)


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
            bus_volt = np.array(env.debug_data[f'iEEE_9.B{bus}.V'])-1.0
            volt_norm += [bus_volt]
            ax[i][j].plot(env.debug_data['iEEE_9.my_time'], bus_volt, color='r')
            ax[i][j].set_ylabel('Voltage Dev')
            ax[i][j].set_title(f'Bus {bus}')

    legend += ['RL Agent']
    gen = ['G2.gENROU.P','G3.gENROU.P']
    for j in range(2):
        ax[3][j].plot(env.debug_data['iEEE_9.my_time'], env.debug_data[f'iEEE_9.{gen[j]}'], color='r')
        ax[3][j].set_xlabel('Time (sec)')
        ax[3][j].set_title(f'{gen[j]} Output')
        ax[4][j].plot(np.arange(250), actions[j::2], color='r')
    ax[3][2].plot(env.debug_data['iEEE_9.my_time'],np.divide(np.cumsum(np.linalg.norm(volt_norm, axis=0)),np.clip(env.debug_data['iEEE_9.my_time'], 1, np.inf)), color='r')
    ax[3][2].plot(env.debug_data['iEEE_9.my_time'],np.linalg.norm(volt_norm, axis=0), color='r')
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
            bus_volt = np.array(env.debug_data[f'iEEE_9.B{bus}.V'])-1.0
            volt_norm += [bus_volt]
            ax[i][j].plot(env.debug_data['iEEE_9.my_time'], bus_volt, color='b')
    legend += ['Randomized']
    for j in range(2):
        ax[3][j].plot(env.debug_data['iEEE_9.my_time'], env.debug_data[f'iEEE_9.{gen[j]}'], color='b')
        ax[4][j].plot(np.arange(250), actions[j::2], color='b')
    ax[3][2].plot(env.debug_data['iEEE_9.my_time'],np.linalg.norm(volt_norm, axis=0), color='b')
    ax[3][2].plot(env.debug_data['iEEE_9.my_time'],np.divide(np.cumsum(np.linalg.norm(volt_norm, axis=0)),np.clip(env.debug_data['iEEE_9.my_time'], 1, np.inf)), color='b')
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
            bus_volt = np.array(env.debug_data[f'iEEE_9.B{bus}.V'])-1.0
            volt_norm += [bus_volt]
            ax[i][j].plot(env.debug_data['iEEE_9.my_time'], bus_volt, color='g')
    legend += ['Do Nothing']
    for j in range(2):
        ax[3][j].plot(env.debug_data['iEEE_9.my_time'], env.debug_data[f'iEEE_9.{gen[j]}'], color='g')
        ax[4][j].plot(np.arange(250), actions[j::2], color='g')
    ax[3][2].plot(env.debug_data['iEEE_9.my_time'],np.divide(np.cumsum(np.linalg.norm(volt_norm, axis=0)),np.clip(env.debug_data['iEEE_9.my_time'], 1, np.inf)), color='g')
    ax[3][2].plot(env.debug_data['iEEE_9.my_time'],np.linalg.norm(volt_norm, axis=0), color='g')
    ax[0][2].legend(legend)
    ax[4][2].plot(np.arange(250), rewards, color='g')

    env.dymola.close()

    plt.savefig(model_name)
    plt.show()
    return

def train_model(env, model_name):
    model = SAC(MlpPolicy, env, learning_rate=10**-3, ent_coef=0.0001, verbose=1, tensorboard_log='tensorboard_log', batch_size=64)
    # model = SAC.load('v3_wInertia', env=env)
    env.reset()
    for _ in range(10):
        print("this", os.getcwd())
        model.learn(1000, reset_num_timesteps=False)
        model.save(model_name)
    print("*******************")
    print(env.total_run_time, env.total_simulation_time)
    return

if __name__=='__main__':
    # model_name = "v5"
    model_name = 'v25'

    # these config values are passed to the model specific environment class
    # mo_name and libs are passed on to the DymolaBaseEnv class
    config = {
        'mo_name': "IEEE9.IEEE_9_RL",
        'libs': ["../../../OpenIPSL-1.5.0/OpenIPSL/package.mo",
               "../../../OpenIPSL-1.5.0/ApplicationExamples/IEEE9/package.mo"],
        'time_step': 60,
        'positive_reward': 1, # unused
        'negative_reward': -100, # unused
        'log_level': 0,
        'default_action': [0,0],
        'method':'Dassl'
    }

    # enable the model specific class as an OpenAI gym environment
    from gym.envs.registration import register
    env_name = "MicrogridEnv-v0"
    register(
        id=env_name,
        entry_point='examples:IEEE9Env',
        kwargs=config
    )

    # create the environment. this will run an initial step and must return [True, [...]] or something is broken
    # TODO: create error handling/warnings if simulations don't work (i.e. returns [False], [...])
    print('making the env')
    env = gym.make(env_name)

    print('calling train')
    train_model(env, model_name)

    print('calling evaluate')
    eval_model(env, model_name)
