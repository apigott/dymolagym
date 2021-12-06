from gym.envs.registration import register
env_name = "MyEnv-v0"
conf_file = "config.json"
register(
    id=env_name,
    entry_point='modelicagym:MyEnv',
    kwargs={"config_file":conf_file}
)

from .eval import *
from .env import *
