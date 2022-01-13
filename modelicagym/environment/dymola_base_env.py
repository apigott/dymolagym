import datetime
import logging
import os
import gym
from gym import spaces
from dymola.dymola_interface import DymolaInterface
import numpy as np
from enum import Enum
import time
import concurrent.futures as futures
import psutil
import DyMat
import json

logger = logging.getLogger(__name__)

"""
Notes from sergio:
render function -- add capability to open the model in dymola
offline check of results to determine/constrain the action space
possibly need to change the injection to occur at startTime+0.1 seconds rather than startTime (use a step block)
    look into how the OpenModelica gym repo did this

"""

def flatten(state):
    return state

def timeout(timelimit):
    def decorator(func):
        def decorated(*args, **kwargs):
            with futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    result = future.result(timelimit)
                except futures.TimeoutError:
                    print('Time out!')
                    result= None
                executor._threads.clear()
                futures.thread._threads_queues.clear()
                return result
        return decorated
    return decorator

class DymolaBaseEnv(gym.Env):
    """
    A variation of the ModelicaGym FMU interface that uses the Dymola API.

    To install dymola API add dymola.exe to the system PATH variable.
    """

    def __init__(self):
        """
        :param model_path: path to the model FMU. Absolute path is advised.
        :param mode: FMU exporting mode "CS" or "ME"
        :param config: dictionary with model specifications:
            model_input_names - names of parameters to be used as action.
            model_output_names - names of parameters to be used as state descriptors.
            model_parameters - dictionary of initial parameters of the model
            time_step - time difference between simulation steps

            positive_reward - (optional) positive reward for default reward policy. Is returned when episode goes on.
            negative_reward - (optional) negative reward for default reward policy. Is returned when episode is ended
        :param log_level: level of logging to be used
        """
        with open(self.conf_file) as f:
            config = json.load(f)
        logger.setLevel(config['log_level'])

        self.model_name = config['mo_name']
        self.dymola = None
        self.libs = config['libs']
        self.reset_dymola()

        # Parameters required by this implementation
        self.tau = config['time_step']
        self.model_input_names = []
        for i in range(len(config['action_names'])):
            if config['action_len'][i] <= 1:
                self.model_input_names += [config['action_names'][i]]
            else:
                self.model_input_names += [f"{config['action_names'][i]}[{x}]" for x in range(1,1+config['action_len'][i])]
        self.model_output_names = config['state_names']
        self.model_parameters = config['model_parameters']
        self.default_action = config['default_action']
        self.add_names = config['additional_debug_states']
        self.method = config['method']
        self.negative_reward = config['negative_reward']
        self.positive_reward = config['positive_reward']
        if not len(self.model_input_names) == len(self.default_action):
            raise Exception("The number of action names and the default action values should be the same length.")
        # initialize the model time and state
        self.start = 0
        self.stop = self.tau
        self.done = False
        self.state = self.reset()

        # OpenAI Gym requirements
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self.debug_data = {name:[] for name in self.model_output_names}
        self.rbc_action_names = []
        for i in range(len(config['rbc_action_names'])):
            if config['rbc_action_len'][i] <= 1:
                self.rbc_action_names += [config['rbc_action_names'][i]]
            else:
                foo = config['rbc_action_names'][i]
                bar = config['rbc_action_len'][i]
                self.rbc_action_names += [f"{foo}[{x}]" for x in range(1,1+bar)]
        self.rbc_action = []
        self.data = None
        self.tracker = 0
        self.exception_flag = False

    # OpenAI Gym API
    def render(self, **kwargs):
        """
        OpenAI Gym API. Determines how current environment state should be rendered.
        :param kwargs:
        :return: implementation should return rendering result
        """
        pass

    def reset(self):
        """
        OpenAI Gym API. Determines restart procedure of the environment
        :return: environment state after restart
        """
        logger.info("Resetting the environment and deleting old results files.")
        if os.path.isdir('temp_dir'):
            # print("Removing old files...")
            for file in os.listdir('temp_dir'):
                try:
                    os.remove(os.path.join(os.getcwd(), 'temp_dir',file))
                except:
                    pass

        self.action = self.default_action
        self.start = 0
        self.stop = self.tau

        logger.info("The model is being reset")
        res = self.dymola.simulateExtendedModel(self.model_name,
                                                startTime=self.start, stopTime=self.stop,
                                                initialNames=self.model_input_names,
                                                initialValues=self.action,
                                                finalNames=self.model_output_names)

        self.state = res[1]
        self.cached_values = None
        self.cached_state = None
        self.debug_data = {name:[] for name in self.model_output_names+self.add_names}
        self.state = self.postprocess_state(self.state)
        self.done = not(res[0])

        if self.done:
            logger.error(f"Model failed to reset. Dymola simulation from time {self.start} to {self.stop} failed with error message: {self.dymola.getLastError()}")
        return flatten(self.state)

    def reset_dymola(self):
        if self.dymola: # this doesn't really seem to be working. It hangs
            self.dymola.close()

        self.dymola = DymolaInterface()
        self.dymola.ExecuteCommand("Advanced.Define.DAEsolver = true")

        # load libraries
        loaded = []
        for lib in self.libs: # all paths relative to the cwd
            loaded += [self.dymola.openModel(lib, changeDirectory=False)]
            if not loaded[-1]:
                logger.error(f"Could not find library {lib}")

        if not False in loaded:
            logger.debug("Successfully loaded all libraries.")

        if not os.path.isdir('temp_dir'):
            os.mkdir('temp_dir')
        self.temp_dir = os.path.join(os.getcwd(), "temp_dir")
        self.dymola.cd('temp_dir')
        logger.debug("Dymola has been reset")
        return

    def step(self, action):
        """
        OpenAI Gym API. Determines how one simulation step is performed for the environment.
        Simulation step is execution of the given action in a current state of the environment.
        :param action: action to be executed.
        :return: resulting state
        """
        logger.debug("Experiment next step was called.")
        if self.done:
            logger.warning(
                """You are calling 'step()' even though this environment has already returned done = True.
                You should always call 'reset' once you receive 'done = True' -- any further steps are
                undefined behavior.""")
            return np.array(self.state), self.negative_reward, self.done, {}

        # check if action is a list. If not - create list of length 1
        try:
            iter(action)
        except TypeError:
            action = [action]
            logger.warning("Model input values (action) should be passed as a list")

        # Check if number of model inputs equals number of values passed
        if len(action) != len(list(self.model_input_names)):
            message = "List of values for model inputs should be of the length {}," \
                      "equal to the number of model inputs. Actual length {}".format(
                len(list(self.model_input_names)), len(action))
            logger.error(message)
            raise ValueError(message)

        # Set input values of the model
        logger.debug("model input: {}, values: {}".format(self.model_input_names, action))
        try:
            self.done, state = self.do_simulation()
            self.state = state
            self.start += self.tau
            self.stop += self.tau
        except:
            print("exception")
            self.reset()
            self.exception_flag = True

        return flatten(self.state), self._reward_policy(), self.done, {}

    # logging
    def get_log_file_name(self):
        log_date = datetime.datetime.utcnow()
        log_file_name = "{}-{}-{}_{}.txt".format(log_date.year, log_date.month, log_date.day, self.model_name)
        return log_file_name

    # internal logic
    def _get_action_space(self):
        """
        Returns action space according to OpenAI Gym API requirements.

        :return: one of gym.spaces classes that describes action space according to environment specifications.
        """
        low = -1*np.ones(len(self.model_input_names))
        high = 1*np.ones(len(self.model_input_names))
        return spaces.Box(low, high)

    def _get_observation_space(self):
        """
        Returns state space according to OpenAI Gym API requirements.

        :return: one of gym.spaces classes that describes state space according to environment specifications.
        """
        low = -1*np.ones(len(self.model_output_names))
        high = 1*np.ones(len(self.model_output_names))
        return spaces.Box(low, high)

    def _is_done(self, results):
        """
        Determines logic when experiment is considered to be done.

        :return: boolean flag if current state of the environment indicates that experiment has ended.
        """
        return not results[0]

    def do_simulation(self):
        """
        Executes simulation by FMU in the time interval [start_time; stop_time]
        currently saved in the environment.

        :return: resulting state of the environment.
        """
        #print("============================")
        logger.debug("Simulation started for time interval {}-{}".format(self.start, self.stop))
        self.act = self.action + self.rbc_action #self.debug_points[-1*self.n_points:]

        found = self.dymola.importInitialResult('dsres.mat', atTime=self.start)
        x = self.dymola.simulateExtendedModel(self.model_name, startTime=self.start,
                                    stopTime=self.stop,
                                    initialNames=self.model_input_names + self.rbc_action_names,
                                    initialValues = self.act,
                                    finalNames=self.model_output_names)

        finished, state = x

        if finished == False:
            print('finished = False')
            print(self.dymola.getLastError())
            state = self.reset()
            finished = True

        self.get_state_values()
        logger.debug("Simulation results: {}".format(state))
        return not finished, state

    def _reward_policy(self):
        """
        Determines reward based on the current environment state.
        By default, implements simple logic of penalizing for experiment end and rewarding each step.

        :return: reward associated with the current state
        """
        if self.exception_flag:
            self.exception_flag = False
        if self.done:
            reward = self.negative_reward
        else:
            reward = self.positive_reward
        return reward

    def get_state_values(self):
        """
        Extracts the values of model outputs at the end of modeling time interval from simulation result

        :return: Values of model outputs as tuple in order specified in `model_outputs` attribute
        """

        self.data = DyMat.DyMatFile('temp_dir/dsres.mat')
        for name in self.debug_data :
            self.debug_data[name] += self.data[name].tolist()

        for name in self.add_names + self.model_input_names + self.rbc_action_names:
            if not name in self.debug_data:
                self.debug_data.update({name:[]})
                self.debug_data[name] += self.data[name].tolist()

        return

    def _set_init_parameter(self):
        """
        Sets initial parameters of a model.

        :return: environment
        """
        if self.model_parameters is not None:
            self.model.set(list(self.model_parameters),
                           list(self.model_parameters.values()))
        return self
