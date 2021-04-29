import datetime
import logging
import os
import gym
from dymola.dymola_interface import DymolaInterface
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

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

        self.model_name = mo_name #model_path.split(os.path.sep)[-1]
        self.dymola = DymolaInterface()

        # load libraries
        loaded = []
        for lib in libs: # all paths relative to the cwd
            loaded += [self.dymola.openModel(lib)]
            self.dymola.cd(os.getcwd()) # return dymola cwd to system cwd

        if not False in loaded:
            logger.debug("Successfully libraries")

        if not os.path.isdir('temp_dir'):
            os.mkdir('temp_dir')
        self.temp_dir = os.path.join(os.getcwd(), "temp_dir")
        self.dymola.cd('temp_dir')

        # if you reward policy is different from just reward/penalty - implement custom step method
        self.positive_reward = config.get('positive_reward')
        self.negative_reward = config.get('negative_reward')

        # Parameters required by this implementation
        self.tau = config.get('time_step')
        self.model_input_names = config.get('model_input_names')
        self.model_output_names = config.get('model_output_names')
        self.model_parameters = config.get('model_parameters')
        self.default_action = config['default_action']

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
        logger.info("Resetting the environment by deleting old results files.")
        if os.path.isdir('temp_dir'):
            # print("Removing old files...")
            for file in os.listdir('temp_dir'):
                os.remove(os.path.join(os.getcwd(), 'temp_dir',file))

        self.action = self.default_action
        self.start = 0
        self.stop = self.tau
        res = self.dymola.simulateExtendedModel(self.model_name,
                                                startTime=self.start, stopTime=self.start,
                                                initialNames=self.model_input_names,
                                                initialValues=self.action,
                                                finalNames=self.model_output_names)
        self.state = res[1]
        return self.state

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

        # Simulate and observe result state
        self.done, self.state = self.do_simulation()

        # Check if experiment has finished
        # self.done = self._is_done()

        # Move simulation time interval if experiment continues
        if not self.done:
            logger.debug("Experiment step done, experiment continues.")
            self.start += self.tau
            self.stop += self.tau
        else:
            logger.WARN("Experiment step done, SIMULATION FAILED.")

        return self.state, self._reward_policy(), self.done, {}

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
    def do_simulation(self):
        """
        Executes simulation by FMU in the time interval [start_time; stop_time]
        currently saved in the environment.

        :return: resulting state of the environment.
        """
        logger.debug("Simulation started for time interval {}-{}".format(self.start, self.stop))

        self.dymola.importInitialResult('dsres.mat', atTime=self.start)

        res = self.dymola.simulateExtendedModel(self.model_name, startTime=self.start,
                                                stopTime=self.stop,
                                                initialNames=self.model_input_names,
                                                initialValues=self.action,
                                                finalNames=self.model_output_names)

        logger.debug("Simulation results: {}".format(res))
        return self._is_done(res), self.get_state(res) # a list of the final values

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
