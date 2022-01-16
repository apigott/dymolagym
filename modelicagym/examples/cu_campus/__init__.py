from gym.envs.registration import register
env_name = "MicrogridEnv-v2"
conf_file = "config.json"
register(
    id=env_name,
    entry_point='modelicagym:CampusEnv',
    kwargs={"config_file":conf_file}
)

from .eval import *
from .env import *
