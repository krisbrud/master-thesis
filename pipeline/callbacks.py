from typing import Dict, Tuple, Union
import argparse
import numpy as np
import os

import ray
from ray.tune.logger import DEFAULT_LOGGERS
import ray.air.callbacks.wandb

# from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy

# from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID


class GymAuvCallbacks(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        **kwargs
    ) -> None:
        # Make a metric of the progress along the path
        path_progress = worker.env.env.progress
        episode.custom_metrics["path_progress"] = path_progress

        # Make a metric of the mean and std of actions
        mean_actions = worker.env.env.episode_action_mean
        mean_episode_throttle = mean_actions[0]
        mean_episode_rudder = mean_actions[1]
        episode.custom_metrics["mean_throttle"] = mean_episode_throttle
        episode.custom_metrics["mean_episode_rudder"] = mean_episode_rudder

        std_actions = worker.env.env.episode_action_std
        std_episode_throttle = std_actions[0]
        std_episode_rudder = std_actions[1]
        episode.custom_metrics["std_throttle"] = std_episode_throttle
        episode.custom_metrics["std_episode_rudder"] = std_episode_rudder

        # Make a metric considering if there was a collision or not
        collision = int(worker.env.env.collision)
        episode.custom_metrics["collision"] = collision


def get_wandb_logger_callback():
    # These to be set as environment variable, for instance in .profile
    # or passed as arguments if runnign as container
    project = os.environ["WANDB_PROJECT"]
    api_key = os.environ["WANDB_API_KEY"]

    wandb_logger_callback = ray.air.callbacks.wandb.WandbLoggerCallback(
        project=project, api_key=api_key, log_config=True
    )

    return wandb_logger_callback
