import dymola
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import logging
import gym
import numpy as np
import os
import matplotlib.pyplot as plt
import DyMat
import modelicagym
import modelicagym.examples

env_name = "MicrogridEnvPyPF-v0"

# create the environment. this will run an initial step and must return [True, [...]] or something is broken
# TODO: create error handling/warnings if simulations don't work (i.e. returns [False], [...])
env = gym.make(env_name)

env.reset()

env.dymola.importInitialResult('dsres.mat', atTime=0)

env.dymola.simulateExtendedModel(env.model_name, startTime=env.start,
                                    stopTime=env.stop,
                                    finalNames=env.model_output_names)
env.dymola.getLastErrorLog()
