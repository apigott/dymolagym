"""
Classic cart-pole example implemented with an FMU simulating a cart-pole system.
Implementation inspired by OpenAI Gym examples:
https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""

import logging
import math
import numpy as np
from gym import spaces
from modelicagym.environment import FMI2CSEnv, FMI1CSEnv, FMI2MEEnv

logger = logging.getLogger(__name__)

class CHPEnv(FMI2MEEnv):
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

    def __init__(self, vref_2, vref_3, vref_6, vref_8, time_step, positive_reward,
                 negative_reward, log_level, path):

        logger.setLevel(log_level)

        self.vref_2 = vref_2
        self.vref_3 = vref_3
        self.vref_6 = vref_6
        self.vref_8 = vref_8

        self.viewer = None
        self.display = None
        self.pole_transform = None
        self.cart_transform = None

        config = {
            'model_input_names': ['vref_2'], # leave out other 3 buses for now.
            'model_output_names': ['B1.V','B2.V','B3.V','B4.V','B5.V','B6.V','B7.V','B8.V','B9.V','B10.V','B11.V','B12.V','B13.V','B14.V'],
            'model_parameters': {},
            'initial_state': (1,1,1,1,1,1,1,1,1,1,1,1,1,1),
            'time_step': time_step,
            'positive_reward': positive_reward,
            'negative_reward': negative_reward
        }

        self.n_steps = 0
        # self._normalize_reward()
        self.max_reward = 1
        self.min_reward = 0
        self.avg_reward = 1
        super().__init__(path, config, log_level)

    # def _normalize_reward(self):
    #     self.max_reward = -np.inf
    #     self.min_reward = np.inf
    #     self.avg_reward = 0
    #
    #     obs = self.reset()
    #     for _ in range(30):
    #         action = [1.0]
    #         obs, reward, done, info = self.step(action)
    #         self.avg_reward += (1/30)*reward
    #         if reward > self.max_reward:
    #             self.max_reward = reward
    #         if reward < self.min_reward:
    #             self.min_reward = reward

    def _is_done(self):
        if self.n_steps > 10:
            done = True
            self.n_steps = 0
        else:
            done = False
        return done

    def _get_action_space(self):
        low = np.array([-1,-1])
        high = np.array([1,1])
        return spaces.Box(low, high)

    def _get_observation_space(self):
        low = 0*np.ones(5)
        high = 100*np.ones(5)
        return spaces.Box(low, high)

    def _reward_policy(self):
        return np.sum(self.state)

    def step(self, action):
        self.n_steps += 1
        return super().step(action)

    def render(self, mode='human', close=False):
        return

    def close(self):
        return self.render(close=True)
