from typing import Callable, Union
from ray.rllib.algorithms.dreamer import Dreamer, DreamerConfig
import gym

from models.auv_dreamer_model import AuvDreamerModel


def _get_auv_dreamer_model_options() -> dict:
    model_config = {
        "custom_model": AuvDreamerModel,
        # RSSM/PlaNET parameters
        "deter_size": 200,
        "stoch_size": 30,
        # CNN Decoder Encoder
        "depth_size": 32,
        # General Network Parameters
        "hidden_size": 400,
        # Action STD
        "action_init_std": 5.0,
    }
    return model_config


def auv_dreamer_factory(env_name: str, env_config: Union[dict, None] = None) -> Dreamer:
    """Instantiates the Dreamer algorithm with a default
    model suitable for the gym-auv environment"""
    # Instantiate the config
    dreamer_config = DreamerConfig()
    dreamer_config.framework(framework="torch")

    # Use the specified environment and environment config
    dreamer_config.env = env_name
    if env_config is not None:
        dreamer_config.env_config

    # Use the custom model
    model_options = _get_auv_dreamer_model_options()
    dreamer_config.training(dreamer_model=model_options)

    # Instantiate the algorithm
    dreamer = Dreamer(config=dreamer_config)
    return dreamer


if __name__ == "__main__":
    auv_dreamer_factory()
