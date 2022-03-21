"""
Classic cart-pole example implemented with an FMU simulating a cart-pole system.
Implementation inspired by OpenAI Gym examples:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""

import logging
import math
import numpy as np
from gym import spaces
from modelicagym.environment import DymolaBaseEnv
import pandas as pd

logger = logging.getLogger(__name__)

class CartPole(DymolaBaseEnv):
    """
    Wrapper class for the creation of a custom Modelica based environment (using the Dymola API)
    Wrapper class for creation of cart-pole environment using JModelica-compiled FMU (FMI standard v.2.0).

    Attributes:
        (1) config_file: a .json file that contains specifications for the RL agent and environment

    """

    def __init__(self, config_file):
        self.conf_file = config_file
        super().__init__()

    def postprocess_state(self, state):
        return state

    def _reward_policy(self):
        if self.done:
            reward = self.negative_reward
        else:
            reward = self.positive_reward
        return reward
