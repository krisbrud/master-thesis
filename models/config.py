from typing import Union
import gym_auv
from models.auv_dreamer_model import AuvDreamerModel
from ray.rllib.algorithms.dreamer.dreamer import DreamerConfig


def _get_auv_dreamer_model_options(config: gym_auv.Config) -> dict:
    model_config = {
        "custom_model": AuvDreamerModel,
        # "custom_model_config": {
        "dense_size": config.vessel.dense_observation_size,
        "lidar_shape": config.vessel.lidar_shape,
        # },
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


def get_auv_dreamer_config(
    env_name: str, gym_auv_config: gym_auv.Config) -> DreamerConfig:
    # Instantiate the config
    dreamer_config = DreamerConfig()
    dreamer_config.framework(framework="torch")

    # Use the specified environment and environment config
    dreamer_config.env = env_name

    dreamer_config.batch_size = 100

    dreamer_config.environment(env_config=env_config)

    # Decrease batch size (and learning rates)
    dreamer_config.batch_size = 10
    dreamer_config.td_model_lr /= 5
    dreamer_config.actor_lr /= 5
    dreamer_config.critic_lr /= 5
    dreamer_config.horizon = gym_auv.DEFAULT_CONFIG.episode.max_timesteps

    # Use the custom model
    model_options = _get_auv_dreamer_model_options()
    dreamer_config.training(dreamer_model=model_options)

    return dreamer_config


def get_auv_dreamer_config_dict(env_name: str, gym_auv_config: gym_auv.Config) -> dict:
    # Instantiate the config
    env_config = {"config": gym_auv_config}
    dreamer_config = {
        "framework": "torch",
        # Use the specified environment and environment config
        "env": env_name,
        "batch_size": 10,
        "batch_length": 50,
        # Use the custom model
        "dreamer_model": _get_auv_dreamer_model_options(gym_auv_config),
        # "record_env": True,
        "prefill_timesteps": 0,
        "evaluation_duration": 1,
        "render_env": True,
        "evaluation_config": {
            "render_env": True,
        },
        # "monitor": True,
    }
    # if env_config is not None:
    #     dreamer_config["env_config"] = env_config
    #     dreamer_config["evaluation_config"]["env_config"] = {
    #         "config": env_config,
    #         "render_mode": "rgb_array",
    #     }

    return dreamer_config
