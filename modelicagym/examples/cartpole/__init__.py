from gym.envs.registration import register
env_name = "CartPoleMo-v0"
conf_file = "config.json"

try:
    register(
        id=env_name,
        entry_point='modelicagym.examples:CartPole',
        kwargs={"config_file":conf_file}
    )
except:
    print("Warning: Environment was not re-registered. If you would like to make changes to this environment you must clear the kernel and reload the library first.")

from .eval import *
from .env import *
