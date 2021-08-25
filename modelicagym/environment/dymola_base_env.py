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
import DyMat

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
        self.add_names = config['additional_debug_states']
        self.method = config['method']
        self.fixedstepsize = None

        # initialize the model time and state
        self.start = 0
        self.stop = self.tau
        self.done = False
        self.state = self.reset()

        # OpenAI Gym requirements
        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self.debug_data = {name:[] for name in self.model_output_names}


        self.rbc_action_names = config.get('model_rbc_names')
        self.rbc_action = []
        self.data = None
        # self.csv_names = []#config.get('csv_names')
        # self.csv_values = []
        self.tracker = 0
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

        print('resetting')
        res = self.dymola.simulateExtendedModel(self.model_name,
                                                startTime=self.start, stopTime=self.stop,
                                                initialNames=self.model_input_names,
                                                initialValues=self.action,
                                                finalNames=self.model_output_names)

        print(f'reset, {res}')
        self.state = res[1]
        self.cached_values = None
        self.cached_state = None
        self.debug_data = {name:[] for name in self.model_output_names+self.add_names}
        self.state = self.postprocess_state(self.state)
        self.done = False
        print('the model has been reset')
        return flatten(self.state)

    def reset_dymola(self):
        print('resetting dymola...')
        if self.dymola: # this doesn't really seem to be working. It hangs
            self.dymola.close()
        # PROCNAME = "Dymola.exe"
        # for proc in psutil.process_iter():
        #     if proc.name() == PROCNAME:
        #         proc.kill()

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
                You should always call 'reset' once you receive 'done = True' -- any further steps are
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

        self.done, state = self.do_simulation()
        self.state = state
        self.start += self.tau
        self.stop += self.tau

        self.step_end = time.time()
        self.total_simulation_time += self.step_end - self.step_start

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

    def do_simulation(self):
        """
        Executes simulation by FMU in the time interval [start_time; stop_time]
        currently saved in the environment.

        :return: resulting state of the environment.
        """
        #print("============================")
        logger.debug("Simulation started for time interval {}-{}".format(self.start, self.stop))
        sim_start = time.time()
        self.act = self.action + self.debug_points[-1*self.n_points:]
        #try:
        #print(f'importing results at {self.start}')
        found = self.dymola.importInitialResult('dsres.mat', atTime=self.start)
        # if found:
            #print(f'simulating at {self.start}')
        x = self.dymola.simulateExtendedModel(self.model_name, startTime=self.start,
                                    stopTime=self.stop,
                                    initialNames=self.model_input_names + self.rbc_action_names,
                                    initialValues = self.act,
                                    finalNames=self.model_output_names)
        #print(x)
        finished, state = x

        if finished == False:
            print('finished = False')
            print(self.dymola.getLastError())
            state = self.reset()
            finished = True

        self.get_state_values()
        logger.debug("Simulation results: {}".format(state))
        sim_end = time.time()
        self.total_simulation_time += sim_end - sim_start
        return not finished, state # a list of the final values

    def _reward_policy(self):
        """
        Determines reward based on the current environment state.
        By default, implements simple logic of penalizing for experiment end and rewarding each step.

        :return: reward associated with the current state
        """
        if self.done:
            reward = -100
        else:
            reward = 1
        return reward

    def get_state_values(self):
        """
        Extracts the values of model outputs at the end of modeling time interval from simulation result

        :return: Values of model outputs as tuple in order specified in `model_outputs` attribute
        """

        self.data = DyMat.DyMatFile('temp_dir/dsres.mat')
        for name in self.debug_data :
            self.debug_data[name] += self.data[name].tolist()

        for name in self.add_names:
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
