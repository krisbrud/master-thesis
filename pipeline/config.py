from typing import Union
import gym_auv
from ray.rllib.algorithms.dreamer.dreamer import DreamerConfig
from ray import tune
from models.auv_dreamer_model import AuvDreamerModel
from pipeline.callbacks import GymAuvCallbacks


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
        "use_lidar": config.vessel.use_lidar,
    }
    return model_config


def get_auv_dreamer_config_dict(env_name: str, gym_auv_config: gym_auv.Config) -> dict:
    # Instantiate the config
    env_config = {"config": gym_auv_config}
    dreamer_config = {
        "framework": "torch",
        # Use the specified environment and environment config
        "env": env_name,
        "batch_size": 50,
        "batch_length": 50,
        # "rollout_fragment_length": 10e3,
        "normalize_actions": False,
        "callbacks": GymAuvCallbacks,
        # Use the custom model
        "dreamer_model": _get_auv_dreamer_model_options(gym_auv_config),
        # "record_env": True,
        "prefill_timesteps": 10e3, 
        "evaluation_duration": 1,
        "render_env": False,
        "evaluation_config": {
            "render_env": True,
        },
        "gamma": 0.99,
        "explore_noise": 1e-3,
        "free_nats": 3,
    }
    return dreamer_config


def get_ray_tune_auv_dreamer_config(env_name: str, gym_auv_config: gym_auv.Config) -> dict:
    # Instantiate the config
    env_config = {"config": gym_auv_config}

    model_options = {
        "custom_model": AuvDreamerModel,
        # "custom_model_config": {
        "dense_size": gym_auv_config.vessel.dense_observation_size,
        "lidar_shape": gym_auv_config.vessel.lidar_shape,
        # },
        # RSSM/PlaNET parameters
        "deter_size": tune.randint(6, 200),
        "stoch_size": tune.randint(1, 30),
        # CNN Decoder Encoder
        "depth_size": 32,
        # General Network Parameters
        "hidden_size": tune.randint(6, 500),
        # Action STD
        "action_init_std": 5.0,
        "use_lidar": gym_auv_config.vessel.use_lidar,
    }

    dreamer_config = {
        "framework": "torch",
        # Use the specified environment and environment config
        "env": env_name,
        "batch_size": 50,
        "batch_length": 50,
        "td_model_lr": tune.loguniform(1e-4, 5e-3),
        "actor_lr": tune.loguniform(1e-5, 5e-4),
        "critic_lr": tune.loguniform(1e-5, 5e-4),
        "grad_clip": tune.randint(50, 200),
        # "rollout_fragment_length": 10e3,
        "normalize_actions": tune.choice([True, False]),
        "callbacks": GymAuvCallbacks,
        # Use the custom model
        "dreamer_model": _get_auv_dreamer_model_options(gym_auv_config),
        # "record_env": True,
        "prefill_timesteps": tune.choice([10e3, 50e3, 100e3]), 
        "evaluation_duration": 5,
        "evaluation_duration_unit": "episodes",
        "render_env": False,
        # ""
        # "evaluation_config": {
        #     "render_env": True,
        # },
        "gamma": tune.loguniform(0.9, 0.999),
        "explore_noise": tune.loguniform(5e-4, 3e-3),
        "free_nats": tune.loguniform(1e-4, 3),
        # "wandb": {

        # }
    }
    return dreamer_config

