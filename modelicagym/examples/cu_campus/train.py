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

def train_model(env, model_name):
    model = SAC(MlpPolicy, env, learning_rate=10**-5, ent_coef='auto_0.1', verbose=1, tensorboard_log='tensorboard_log', batch_size=64)

    env.reset()
    for _ in range(10):
        model.learn(1000, reset_num_timesteps=False)
        model.save(f"models/{env.model_name}")
    return

if __name__=='__main__':
    env = gym.make(env_name)

    train_model(env)
    eval_model(env)
