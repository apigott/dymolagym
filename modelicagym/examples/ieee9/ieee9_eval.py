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

def eval_ieee9(env, model_name):
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
    fig, ax = plt.subplots(6,3,figsize=(40,30))
    for i in range(3):
        for j in range(3):
            bus = 1 + 3*i + j
            bus_volt = np.array(env.debug_data[f'iEEE_9.B{bus}.V'])-1.0
            volt_norm += [bus_volt]
            ax[i][j].plot(env.debug_data['iEEE_9.my_time'], (bus_volt), color='r')
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
    ax[4][2].plot(rewards, color='r')
    sum_loads = None
    sum_gen = None
    for load in ['iEEE_9.load_B5.P','iEEE_9.load_B6.P','iEEE_9.load_B8.P']:
        if sum_loads is None:
            sum_loads = env.debug_data[load]
        else:
            sum_loads = np.add(sum_loads,env.debug_data[load])
        ax[5][0].plot(env.debug_data[load],color='r')

    for g in gen + ['G1.P']:
        if sum_gen is None:
            sum_gen = env.debug_data['iEEE_9.'+g]
        else:
            sum_gen = np.add(sum_gen,env.debug_data['iEEE_9.'+g])
        ax[5][1].plot(env.debug_data['iEEE_9.'+g],color='r')
    ax[5][2].plot(sum_gen,color='r')
    ax[5][2].plot(sum_loads,color='r')


    # env.reset()
    # actions = []
    # rewards = []
    # for _ in range(250):
    #     action = env.action_space.sample()
    #     actions += action.tolist()
    #     obs, reward, done, info = env.step(action)
    #     rewards += [reward]
    # volt_norm = []
    # for i in range(3):
    #     for j in range(3):
    #         bus = 1 + 3*i + j
    #         bus_volt = np.array(env.debug_data[f'iEEE_9_wVariation.B{bus}.V'])-1.0
    #         volt_norm += [bus_volt]
    #         ax[i][j].plot(env.debug_data['iEEE_9_wVariation.my_time'], bus_volt, color='b')
    # legend += ['Randomized']
    # for j in range(2):
    #     ax[3][j].plot(env.debug_data['iEEE_9_wVariation.my_time'], env.debug_data[f'iEEE_9_wVariation.{gen[j]}'], color='b')
    #     ax[4][j].plot(np.arange(250), actions[j::2], color='b')
    # ax[3][2].plot(env.debug_data['iEEE_9_wVariation.my_time'],np.linalg.norm(volt_norm, axis=0), color='b')
    # ax[3][2].plot(env.debug_data['iEEE_9_wVariation.my_time'],np.divide(np.cumsum(np.linalg.norm(volt_norm, axis=0)),np.clip(env.debug_data['iEEE_9_wVariation.my_time'], 1, np.inf)), color='b')
    # ax[4][2].plot(np.arange(250), rewards, color='b')

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
            ax[i][j].plot(env.debug_data['iEEE_9.my_time'], (bus_volt), color='g')
    legend += ['Do Nothing']
    for j in range(2):
        ax[3][j].plot(env.debug_data['iEEE_9.my_time'], env.debug_data[f'iEEE_9.{gen[j]}'], color='g')
        ax[4][j].plot(np.arange(250), actions[j::2], color='g')
    ax[3][2].plot(env.debug_data['iEEE_9.my_time'],np.divide(np.cumsum(np.linalg.norm(volt_norm, axis=0)),np.clip(env.debug_data['iEEE_9.my_time'], 1, np.inf)), color='g')
    ax[3][2].plot(env.debug_data['iEEE_9.my_time'],np.linalg.norm(volt_norm, axis=0), color='g')
    ax[0][2].legend(legend)
    ax[4][2].plot(rewards, color='g')
    sum_loads = None
    sum_gen = None
    for load in ['iEEE_9.load_B5.P','iEEE_9.load_B6.P','iEEE_9.load_B8.P']:
        if sum_loads is None:
            sum_loads = env.debug_data[load]
        else:
            sum_loads = np.add(sum_loads,env.debug_data[load])
        ax[5][0].plot(env.debug_data[load],color='g')

    for g in gen + ['G1.P']:
        if sum_gen is None:
            sum_gen = env.debug_data['iEEE_9.'+g]
        else:
            sum_gen = np.add(sum_gen,env.debug_data['iEEE_9.'+g])
        ax[5][1].plot(env.debug_data['iEEE_9.'+g],color='g')
    ax[5][2].plot(sum_gen,color='g')
    ax[5][2].plot(sum_loads,color='g')

    env.dymola.close()

    plt.savefig(model_name)
    plt.show()

if __name__=='__main__':
    model_name = 'penalize_slack'

    # add reference libraries here. Current structure will use the relative path from this file
    libs = ["../../../OpenIPSL-1.5.0/OpenIPSL/package.mo",
           "../../../OpenIPSL-1.5.0/ApplicationExamples/IEEE9/package.mo"]

    # check that all the paths to library package.mo files exist
    # DymolaInterface() also checks this but I've found this warning helpful
    for lib in libs:
        if not os.path.isfile(lib):
            print(f"Cannot find the library {lib}")

    mo_name = "IEEE9.IEEE_9_RL" # name of Modelica model in the Library.Model format
    env_entry_point = 'examples:IEEE9Env' # Python package location of RL environment

    time_step = 60 # time delta in seconds
    positive_reward = 1
    negative_reward = -100 # penalize RL agent for is_done
    log_level = 0
    default_action = [0,0]

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

    eval_ieee9(env, model_name)

    if env.dymola:
        env.dymola.close()
