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

class CampusEnv(DymolaBaseEnv):

    def __init__(self, config_file):
        self.conf_file = config_file
        self.load_profiles = pd.read_csv('loads.csv')
        self.solar = pd.read_csv('coors_pv.csv')
        self.solar['Date & Time'] = pd.to_datetime(self.solar['Date & Time'], format='%m/%d/%Y %H:%M')
        self.solar = self.solar.iloc[(self.solar['Date & Time'].dt.hour).idxmin():]
        super().__init__()

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

    def _reward_policy(self):
        avg_voltages = [(np.average(self.data[n]) -1) for i in range(1,10) for n in self.model_output_names]
        reward = -1*np.linalg.norm(avg_voltages)
        reward = 100*(reward + 0.1)
        self.cached_state = self.state
        if self.exception_flag:
            reward -= 5
            self.exception_flag = False
        return reward

    def step(self, action):
        # # load_profile = load_profiles['3'].iloc[:24].to_list()
        # # load_profile = np.subtract(load_profile, load_profile[0])
        self.rbc_action = []
        start_idx = int(self.start/3600) % len(self.load_profiles)
        stop_idx = int(self.stop/3600) % len(self.load_profiles)

        for x in range(1,9):
            foo = self.load_profiles[f'{x}_est'].iloc[int(self.start/3600):int(self.stop/3600)].to_list()
            bar = self.load_profiles[f'{x}_est'].iloc[0]
            foo = np.subtract(foo, bar).tolist()
            self.rbc_action += foo

        # for i in range(1,1441):
        #     # self.rbc_action_names += [f"irradiance_to_Power.irradiance_profile[i]"]
        #     self.rbc_action += [self.solar['Irradiance [W/m^2]'].iloc[i]]
        # print(self.rbc_action)
        return super().step(action)
