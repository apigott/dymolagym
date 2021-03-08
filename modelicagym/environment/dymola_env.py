from modelicagym.environment import DymolaBaseEnv#, FMIStandardVersion
import logging

logger = logging.getLogger(__name__)


class DymolaEnv(DymolaBaseEnv):
    """
    Wrapper class of ModelicaBaseEnv for convenient creation of environments that utilize
    FMU exported in model exchange mode.
    Should be used as a superclass for all such environments.
    Implements abstract logic, handles different Modelica types (JModelica, Dymola) particularities.

    """

    def __init__(self, model_libs, model_name, config, log_level,
                 simulation_start_time=0):
        """

        :param model_path: path to the model FMU. Absolute path is advised.
        :param config: dictionary with model specifications. For more details see ModelicaBaseEnv docs
        :param fmi_version: version of FMI standard used in FMU compilation.
        :param log_level: level of logging to be used in experiments on environment.
        """
        self.simulation_start_time = simulation_start_time
        # self.fmi_version = fmi_version
        logger.setLevel(log_level)
        super().__init__(model_path, config, log_level)

    def reset(self):
        """
        OpenAI Gym API. Restarts environment and sets it ready for experiments.
        In particular, does the following:
            * resets model
            * sets simulation start time to 0
            * sets initial parameters of the model
            * initializes the model
            * sets environment class attributes, e.g. start and stop time.
        :return: state of the environment after resetting
        """
        logger.debug("Experiment reset was called. Resetting the model.")

        # self.model.reset()
        # if self.fmi_version == FMIStandardVersion.second:
        #     self.model.setup_experiment(start_time=0)

        # self._set_init_parameter()
        # self.model.initialize()

        # get initial state of the model from the fmu
        self.start = 0
        self.stop = self.simulation_start_time
        self.state = self.do_simulation()

        self.start = self.simulation_start_time
        self.stop = self.start + self.tau
        self.done = self._is_done()
        return self.state


# class FMI1MEEnv(ModelicaMEEnv):
#     """
#     Wrapper class.
#     Should be used as a superclass for all environments using FMU exported in co-simulation mode,
#     FMI standard version 1.0.
#     Abstract logic is implemented in parent classes.
#
#     Refer to the ModelicaBaseEnv docs for detailed instructions on own environment implementation.
#     """
#
#     def __init__(self, model_path, config, log_level, simulation_start_time=0):
#         super().__init__(model_path, config, FMIStandardVersion.first, log_level,
#                          simulation_start_time=simulation_start_time)
#
#
# class FMI2MEEnv(ModelicaMEEnv):
#     """
#     Wrapper class.
#     Should be used as a superclass for all environments using FMU exported in co-simulation mode.
#     FMI standard version 2.0.
#     Abstract logic is implemented in parent classes.
#
#     Refer to the ModelicaBaseEnv docs for detailed instructions on own environment implementation.
#     """
#     def __init__(self, model_path, config, log_level, simulation_start_time=0):
#         super().__init__(model_path, config, FMIStandardVersion.second, log_level,
#                          simulation_start_time=simulation_start_time)
