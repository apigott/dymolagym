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
import pandapower as pp

logger = logging.getLogger(__name__)

class CampusEnvPyPF(DymolaBaseEnv):
    def __init__(self, config_file):
        self.conf_file = config_file
        self.load_profiles = pd.read_csv('loads.csv')
        self.solar = pd.read_csv('coors_pv.csv')
        self.solar['Date & Time'] = pd.to_datetime(self.solar['Date & Time'], format='%m/%d/%Y %H:%M')
        self.solar = self.solar.iloc[(self.solar['Date & Time'].dt.hour).idxmin():]
        self.microgrid = self.create_network()
        self.gen_profile = []
        super().__init__()

    def create_network(self):
        microgrid = pp.create_empty_network(name="cub_campus")
        high_voltage = 23
        low_voltage = 13.8
        gen_voltage = 0.4

        # Xcel energy operates at 13.8kV for distribution
        grid_bus1 = pp.create_bus(microgrid, name="GRID_bus1", vn_kv=high_voltage,geodata=(0,0))
        lv_bus1 = pp.create_bus(microgrid, name="LV_bus1", vn_kv=low_voltage, type="n",geodata=(0.5,0))
        sb_bus1 = pp.create_bus(microgrid, name="SB_bus1", vn_kv=low_voltage, type="n",geodata=(1,0))
        multidomain_bus = pp.create_bus(microgrid, name="MultiDomain_bus", vn_kv=gen_voltage, type="n",geodata=(1,0.5))
        multidomain_bus1 = pp.create_bus(microgrid, name="MultiDomain_bus1", vn_kv=low_voltage, type="n", geodata=(1,0.5))
        central_bus = pp.create_bus(microgrid, name="CENTRAL_bus", vn_kv=low_voltage,geodata=(1.5,0))
        pv_bus = pp.create_bus(microgrid, name="PV_bus", vn_kv=low_voltage,geodata=(2,0.5))
        bess_bus = pp.create_bus(microgrid, name="BESS_bus", vn_kv=low_voltage, type="n",geodata=(2,1))
        gen_bus = pp.create_bus(microgrid, name="Gen_bus", vn_kv=low_voltage, type="n",geodata=(0.5,0.5))
        sec1 = pp.create_bus(microgrid, name="sec1", vn_kv=low_voltage, type="n",geodata=(3,1))
        sec2 = pp.create_bus(microgrid, name="sec2", vn_kv=low_voltage, type="n",geodata=(3,-1))
        load1 = pp.create_bus(microgrid, name="macky", vn_kv=low_voltage, type="n",geodata=(3.5,1.3))
        load2 = pp.create_bus(microgrid, name="hellums", vn_kv=low_voltage, type="n",geodata=(3.5,0.7))
        load3 = pp.create_bus(microgrid, name="stadium", vn_kv=low_voltage, type="n",geodata=(3,0))
        load4 = pp.create_bus(microgrid, name="bookstore", vn_kv=low_voltage, type="n",geodata=(4,1))
        load5 = pp.create_bus(microgrid, name="quad", vn_kv=low_voltage, type="n",geodata=(3.5,-0.7))
        load6 = pp.create_bus(microgrid, name="ec", vn_kv=low_voltage, type="n",geodata=(3.5,-1.3))
        load7 = pp.create_bus(microgrid, name="chw_plant", vn_kv=low_voltage, type="n",geodata=(4,-0.7))
        load8 = pp.create_bus(microgrid, name="kitt", vn_kv=low_voltage, type="n",geodata=(4,-1.3))

        buses = [grid_bus1, lv_bus1, sb_bus1, multidomain_bus1, central_bus, pv_bus, bess_bus, sec1, sec2, load1, load2, load3, load4, load5, load6, load7, load8, gen_bus]
        conns = [[1,2],[2,4],[3,4],[4,5],[4,6],[4,7],[4,11],[4,8],[7,9],[7,12],[9,10],[10,12],[8,13],[8,15],[13,14],[14,16],[16,15],[3,17]]
        for conn in conns:
            pp.create_line(microgrid, buses[conn[0]], buses[conn[1]], 0.5, std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV")

        xcel_conn = pp.create_ext_grid(microgrid, grid_bus1, vm_pu=1.02, va_degree=50)
        xcel_transformer = pp.create_transformer_from_parameters(microgrid, grid_bus1, lv_bus1, 2000, high_voltage, low_voltage, 0.25, 10, 0.48, 0.06)
        chp_gen = pp.create_gen(microgrid, gen_bus, p_mw=3.5)
        chp_transformer = pp.create_transformer_from_parameters(microgrid, multidomain_bus, multidomain_bus1, 2000, gen_voltage, low_voltage, 0.25, 10, 0.48, 0.06)
        pv_gen = pp.create_gen(microgrid, pv_bus, p_mw=0)

        ps = [347.648998,189.584781,4337.210930,1923.991871,3396.980054,796.892406,972.234648,1028.719053]
        ls = [load1, load2, load3, load4, load5, load6, load7, load8]
        i = 0
        for l in ls:
            p = ps[i] * 0.001
            s = p * np.tan(np.arccos(np.random.uniform(0.85,0.95)))
            pp.create_load(microgrid, l, name=microgrid.bus.iloc[l].name, p_mw=p, q_mvar=s, const_i_percent=1, const_z_percent=3)
            i += 1
        return microgrid

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

    def update_loads(self):
        for i in self.microgrid.load.index:
            name = self.microgrid.load.name.iloc[i]
            if name == 'chw_plant':
                self.microgrid.load.iloc[i]['p_mw'] = self.data["pumDis.P"]
                self.microgrid.load.iloc[i]['q_mvar'] = self.data["pumDis.P"]*np.tan(np.arccos(0.9))
            else:
                self.microgrid.load.iloc[i]['p_mw'] = np.random.uniform(0.95,1.05)*self.microgrid.load.iloc[i]['p_mw']
                self.microgrid.load.iloc[i]['q_mvar'] = np.random.uniform(0.95, 1.05)*self.microgrid.load.iloc[i]['q_mvar']

    def step(self, action):

        sarsa = super().step(action)

        self.update_loads()
        pp.runpp(self.microgrid)
        self.gen_profile += [[self.microgrid.res_ext_grid.p_mw, self.microgrid.res_ext_grid.q_mvar]]
        return sarsa
