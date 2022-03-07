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

# evaluate() # eval() is a protected function name
def evaluate(env, model_name):
    env.reset()
    env.reset_dymola()

if __name__=='__main__':
    env_name = "MicrogridEnv-v0"

    # create the environment. this will run an initial step and must return [True, [...]] or something is broken
    # TODO: create error handling/warnings if simulations don't work (i.e. returns [False], [...])
    env = gym.make(env_name)

    evaluate(env, env.model_name)

    if env.dymola:
        env.dymola.close()
