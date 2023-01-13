from ray.rllib.algorithms.dreamer import Dreamer, DreamerConfig
import gym_auv

from models.auv_dreamer import AuvDreamer # , auv_dreamer_factory
# from models.auv_dreamer import _get_auv_dreamer_model_options
from pipeline.config import get_auv_dreamer_config_dict
from pipeline.register_components import register_gym_auv_scenarios
from torch import cuda

def test_auv_dreamer_initialization():
    # Register the gym environments in gym_auv so rllib can use them
    register_gym_auv_scenarios()

    env_name = "TestScenario1-v0"
    gym_auv_config = gym_auv.DEFAULT_CONFIG
    dreamer_config = get_auv_dreamer_config_dict(
        env_name=env_name, gym_auv_config = gym_auv_config
    )
    dreamer_config["prefill_timesteps"] = 0  # Avoid expensive environment sampling for this test
    if cuda.is_available():
        dreamer_config["num_gpus"] = 1 
    # Make a Dreamer instance
    dreamer = AuvDreamer(dreamer_config, env=env_name) 


if __name__ == "__main__":
    test_auv_dreamer_initialization()
