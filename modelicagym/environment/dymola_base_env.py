import datetime
import logging
import os
import gym
from dymola.dymola_interface import DymolaInterface
import numpy as np
from enum import Enum
import time
import concurrent.futures as futures
import psutil

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
    # flat_state = []
    # for s in state:
    #     try:
    #         flat_state += s
    #     except:
    #         flat_state += [s]
    # print(flat_state)
    # return flat_state

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

    def __init__(self, mo_name, libs, config, log_level):
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
        logger.setLevel(log_level)

        self.model_name = mo_name
        self.dymola = None
        self.libs = libs
        self.reset_dymola()

        # if you reward policy is different from just reward/penalty - implement custom step method
        self.positive_reward = config.get('positive_reward')
        self.negative_reward = config.get('negative_reward')

        # Parameters required by this implementation
        self.tau = config.get('time_step')
        self.model_input_names = config.get('model_input_names')
        self.model_output_names = config.get('model_output_names')
        self.model_parameters = config.get('model_parameters')
        self.default_action = config['default_action']
        self.method = config['method']

        # initialize the model time and state
        self.start = 0
        self.stop = self.tau
        self.done = False
        self.state = self.reset()

        # OpenAI Gym requirements
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        # self.metadata = {
        #     'render.modes': ['human', 'rgb_array'],
        #     'video.frames_per_second': 50
        # }

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
        print('the model will be reset')
        logger.info("Resetting the environment by deleting old results files.")
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

        res = self.dymola.simulateExtendedModel(self.model_name,
                                                startTime=self.start, stopTime=self.start,
                                                initialNames=self.model_input_names,
                                                initialValues=self.action,
                                                finalNames=self.model_output_names,
                                                method=self.method)

        # print(self.dymola.getLastError())

        self.state = res[1]
        self.cached_values = None
        self.state = self.postprocess_state(self.state)
        self.reset_flag = False

        print('the model has been reset')
        return flatten(self.state)

    def reset_dymola(self):
        print('resetting dymola...')
        # if self.dymola: # this doesn't really seem to be working. It hangs
        #     self.dymola.close()
        PROCNAME = "Dymola.exe"
        for proc in psutil.process_iter():
            if proc.name() == PROCNAME:
                proc.kill()

        self.dymola = DymolaInterface()
        self.dymola.ExecuteCommand("Advanced.Define.DAEsolver = true")

        # load libraries
        loaded = []
        for lib in self.libs: # all paths relative to the cwd
            loaded += [self.dymola.openModel(lib, changeDirectory=False)]

        if not False in loaded:
            logger.debug("Successfully loaded all libraries.")
        else:
            logger.error("Dymola could not find all models.")

        if not os.path.isdir('temp_dir'):
            os.mkdir('temp_dir')
        self.temp_dir = os.path.join(os.getcwd(), "temp_dir")
        self.dymola.cd('temp_dir')
        print('dymola has been reset')
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
            logging.warning(
                """You are calling 'step()' even though this environment has already returned done = True.
                You should always call 'reset()' once you receive 'done = True' -- any further steps are
                undefined behavior.""")
            return np.array(self.state), self.negative_reward, self.done, {}

        # check if action is a list. If not - create list of length 1
        try:
            iter(action)
        except TypeError:
            action = [action]
            logging.warning("Model input values (action) should be passed as a list")

        # Check if number of model inputs equals number of values passed
        if len(action) != len(list(self.model_input_names)):
            message = "List of values for model inputs should be of the length {}," \
                      "equal to the number of model inputs. Actual length {}".format(
                len(list(self.model_input_names)), len(action))
            logging.error(message)
            raise ValueError(message)

        # Set input values of the model
        logger.debug("model input: {}, values: {}".format(self.model_input_names, action))
        # self.model.set(list(self.model_input_names), list(action)) # @akp to fix

        self.action = action

        # Reset if the last timestep failed
        if self.reset_flag:
            self.reset()

        # Simulate and observe result state
        res = self.do_simulation()
        if res:
            self.done, self.state = res # will fail on unpack
        else:
            print('it failed')
            # print(self.dymola.getLastError())
            self.reset_dymola()
            self.reset()
            self.done, self.state = self.do_simulation()
        self.state = self.postprocess_state(self.state)
        # Check if experiment has finished
        # self.done = self._is_done()

        # Move simulation time interval if experiment continues
        if not self.done:
            logger.debug("Experiment step done, experiment continues.")
            self.start += self.tau
            self.stop += self.tau
        else:
            logger.warn("Experiment step done, SIMULATION FAILED.")
            self.reset_flag = True
            self.done = False

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
        pass

    def _get_observation_space(self):
        """
        Returns state space according to OpenAI Gym API requirements.

        :return: one of gym.spaces classes that describes state space according to environment specifications.
        """
        pass

    def _is_done(self, results):
        """
        Determines logic when experiment is considered to be done.

        :return: boolean flag if current state of the environment indicates that experiment has ended.
        """
        return not results[0]

    # part of the step() method extracted for convenience
    @timeout(15)
    def do_simulation(self):
        """
        Executes simulation by FMU in the time interval [start_time; stop_time]
        currently saved in the environment.

        :return: resulting state of the environment.
        """
        logger.debug("Simulation started for time interval {}-{}".format(self.start, self.stop))

        self.dymola.importInitialResult('dsres.mat', atTime=self.start)

        try:
            res = self.dymola.simulateExtendedModel(self.model_name, startTime=self.start,
                                                stopTime=self.stop,
                                                initialNames=self.model_input_names,
                                                initialValues=self.action,
                                                finalNames=self.model_output_names,
                                                method=self.method)

            logger.debug("Simulation results: {}".format(res))
            return self._is_done(res), self.get_state(res) # a list of the final values
        except:
            return None

    def _reward_policy(self):
        """
        Determines reward based on the current environment state.
        By default, implements simple logic of penalizing for experiment end and rewarding each step.

        :return: reward associated with the current state
        """
        return self.negative_reward or -100 if self.done else self.positive_reward or 1

    def get_state(self, result):
        """
        Extracts the values of model outputs at the end of modeling time interval from simulation result

        :return: Values of model outputs as tuple in order specified in `model_outputs` attribute
        """
        return result[1]

    def _set_init_parameter(self):
        """
        Sets initial parameters of a model.

        :return: environment
        """
        if self.model_parameters is not None:
            self.model.set(list(self.model_parameters),
                           list(self.model_parameters.values()))
        return self
