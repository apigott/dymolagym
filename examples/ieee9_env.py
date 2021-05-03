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

logger = logging.getLogger(__name__)

class IEEE9Env(DymolaBaseEnv):
    """
    Wrapper class for creation of cart-pole environment using JModelica-compiled FMU (FMI standard v.2.0).

    Attributes:
        m_cart (float): mass of a cart.

        m_pole (float): mass of a pole.

        theta_0 (float): angle of the pole, when experiment starts.
        It is counted from the positive direction of X-axis. Specified in radians.
        1/2*pi means pole standing straight on the cast.

        theta_dot_0 (float): angle speed of the poles mass center. I.e. how fast pole angle is changing.
        time_step (float): time difference between simulation steps.
        positive_reward (int): positive reward for RL agent.
        negative_reward (int): negative reward for RL agent.
    """

    def __init__(self, mo_name, libs, default_action, time_step, positive_reward,
                 negative_reward, log_level):

        logger.setLevel(log_level)

        self.viewer = None
        self.display = None
        self.pole_transform = None
        self.cart_transform = None

        self.action_names = ['G1.pref.k','G2.pref.k','G3.pref.k']
        self.state_names = ['B1.V','B2.V','B3.V','B4.V','B5.V','B6.V','B7.V','B8.V','B9.V']
        # for a time averaged version:
        # self.state_names = ['b1_average.y', 'b2_average.y', ...]
        config = {
            'model_input_names': self.action_names,
            'model_output_names': self.state_names,
            'model_parameters': {},
            'initial_state': (1),
            'time_step': time_step,
            'positive_reward': positive_reward,
            'negative_reward': negative_reward,
            'default_action': default_action
        }

        self.n_steps = 0
        # self._normalize_reward()
        self.max_reward = 0.5
        self.min_reward = -0.5
        self.avg_reward = 0
        # change this to unpack the dictionary __init__(**config), so that some parameters can have default values
        super().__init__(mo_name, libs, config, log_level)

    # def _is_done(self):
    #     done = False
    #     return done

    def _get_action_space(self):
        low = -0.1*np.ones(len(self.action_names))
        high = 0.1*np.ones(len(self.action_names))
        return spaces.Box(low, high)

    def _get_observation_space(self):
        low = 0.92*np.ones(len(self.state_names))
        high = 1.06*np.ones(len(self.state_names))
        return spaces.Box(low, high)

    def _reward_policy(self):
        reward = -10*np.linalg.norm(np.subtract(self.state,1))
        normalized_reward = (reward - self.avg_reward) / (self.max_reward - self.min_reward)
        return normalized_reward

    def step(self, action):
        self.n_steps += 1
        return super().step(list(action))

    def render(self, mode='human', close=False):
        return

    def close(self):
        return self.render(close=True)
