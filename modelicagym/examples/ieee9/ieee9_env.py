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
import time

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
                 negative_reward, log_level, method):

        self.total_simulation_time = 0
        self.total_run_time = 0

        logger.setLevel(log_level)
        np.random.seed(10)
        self.viewer = None
        self.display = None

        self.action_names = ['m','n']
        self.rbc_action_names = [f'x[{i}]' for i in range(1,62)]+[f'y[{i}]' for i in range(1,62)] + [f'z[{i}]' for i in range(1,62)]
        self.n_points = len(self.rbc_action_names)
        self.debug_points = []

        self.state_names = ['iEEE_9.B1.V',
                            'iEEE_9.B2.V',
                            'iEEE_9.B3.V',
                            'iEEE_9.B4.V',
                            'iEEE_9.B5.V',
                            'iEEE_9.B6.V',
                            'iEEE_9.B7.V',
                            'iEEE_9.B8.V',
                            'iEEE_9.B9.V',
                            'iEEE_9.G2.gENROU.P',
                            'iEEE_9.G3.gENROU.P',
                            'iEEE_9.load_B8.P',
                            'iEEE_9.load_B6.P',
                            'iEEE_9.load_B5.P',
                            'iEEE_9.G1.P']

        config = {
            'model_input_names': self.action_names,
            'model_rbc_names': self.rbc_action_names,
            'model_output_names': self.state_names,
            'model_parameters': {},
            'initial_state': (1),
            'time_step': time_step,
            'positive_reward': positive_reward,
            'negative_reward': negative_reward,
            'default_action': default_action,
            'method': method,
            'additional_debug_states': ['iEEE_9.my_time','iEEE_9.load_B5.interp_value','x[1]','m','n']
        }

        self.n_steps = 0
        self.max_reward = 0.5
        self.min_reward = -0.5
        self.avg_reward = 0
        self.cached_values = None
        self.cached_state = None
        self.debug_data = None
        self.initial_value = None
        self.loads = pd.read_csv('all_loads.csv')

        self.num_samples = len(self.loads)
        self.loads['time'] = self.loads['time'].round()
        self.loads['P1'] = self.loads['P1']/self.loads['P1'].iloc[0]
        self.loads['P2'] = self.loads['P2']/self.loads['P2'].iloc[0]
        self.loads['P3'] = self.loads['P3']/self.loads['P3'].iloc[0]
        self.last_value = None
        self.penalty = 0
        self.random_walk = {'P1':[0],'P2':[0],'P3':[0]}
        self.actions = []
        # change this to unpack the dictionary __init__(**config), so that some parameters can have default values
        super().__init__(mo_name, libs, config, log_level)

    def postprocess_state(self, state):
        processed_states = []
        for s in state:
            if isinstance(s, list): # this is lazy and will only work when the whole list is of integrated values
                if self.cached_values:
                    processed_states += list(np.divide(np.subtract(s, self.cached_values),self.tau))
                    self.cached_values = s # this implementation only works when there is "one" variable that is integrated
                else:
                    processed_states += list(np.divide(s,self.tau))
                    self.cached_values = s
            else:
                processed_states += [s]
        return processed_states

    def _get_action_space(self):
        low = -1*np.ones(len(self.action_names))
        high = np.ones(len(self.action_names))
        return spaces.Box(low, high)

    def _get_observation_space(self):
        low = -1*np.ones(len(self.state_names))
        high = 1*np.ones(len(self.state_names))
        return spaces.Box(low, high)

    def _reward_policy(self):
        avg_voltages = [10*(np.average(self.data[f'iEEE_9.B{i}.V']) -1) for i in range(1,10)]
        reward = -1*np.linalg.norm(avg_voltages)
        reward = reward + 0.5
        self.cached_state = self.state
        #print(reward)
        return reward #+ self.penalty

    def step(self, action):
        self.step_start = time.time()
        self.action = np.multiply(0.1,action).tolist()
        self.actions += [self.action]

        # additional processing of state variables
        volt_violations = False
        if self.data:
            for i in range(1,10):
                if not volt_violations:
                    x = np.average(self.data[f'iEEE_9.B{i}.V'])
                    if x < 0.9 or x > 1.1:
                        volt_violations = True

        times = self.start + np.arange(60)
        sin_times = np.linspace(times[0]*6.28/86400,times[-1]*6.28/86400, 61)
        base = 1+0.1*np.sin(sin_times)
        # additional processing of "action variables"
        for val in ['P1','P2','P3']:
            #random_walk = np.clip(np.cumsum(np.random.uniform(-0.02,0.02,60)),a_min=-0.1, a_max=0.1)
            self.random_walk[val] += (self.random_walk[val][-1] + np.cumsum(np.random.uniform(-0.001,0.001, 61))).tolist()
            x = base + self.random_walk[val][-61:]
            self.debug_points += x.tolist()

        # rule based control action
        self.rbc_action = self.debug_points[-1*self.n_points:]
        self.n_steps += 1

        return super().step(self.action)

    def reset(self):
        print('resetting debug points')
        self.debug_points = []
        self.random_walk = {'P1':[0], 'P2':[0], 'P3':[0]}
        np.random.seed(0)
        return super().reset()

    def render(self, mode='human', close=False):
        return

    def close(self):
        return self.render(close=True)
