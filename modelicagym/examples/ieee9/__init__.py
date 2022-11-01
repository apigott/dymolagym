from gym.envs.registration import register
env_name = "ieee9-v0"
conf_file = "config.json"
register(
    id=env_name,
    entry_point='modelicagym:CampusEnv',
    kwargs={"config_file":conf_file}
)

from .ieee9_eval import *
from .ieee9_env import *
