import argparse

import ray
from ray import tune
from ray import air
from ray.tune.search.bayesopt import BayesOptSearch
from torch import cuda
from ray.rllib.algorithms import Algorithm

import gym_auv

from pipeline import callbacks
from pipeline.config import get_ray_tune_auv_dreamer_config
from pipeline.register_envs import register_gym_auv_scenarios
from models.auv_dreamer import (
    AuvDreamer,
)


def main():
    ray.init()
    # Register environments from gym_auv
    register_gym_auv_scenarios()

    env_name = "MovingObstaclesLosRewarder-v0"

    gym_auv_config = gym_auv.LOS_COLAV_CONFIG
    
    dreamer_config = get_ray_tune_auv_dreamer_config(
        env_name=env_name, gym_auv_config=gym_auv_config
    )

    dreamer_config["prefill_timesteps"] = 1000

    if cuda.is_available():
        # Use GPU if available
        dreamer_config["num_gpus"] = 1

    tuner = tune.Tuner(
        tune.with_resources(
            AuvDreamer,
            {"cpu": 8, "gpu": 1},
        ),
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            num_samples=1,
        ),
        run_config=air.RunConfig(
            stop={"training_iteration": 1},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=1, 
                checkpoint_at_end=True
            )
        ),
        param_space=dreamer_config,
    )
    results = tuner.fit()
    print(results)

    ray.shutdown()


if __name__ == "__main__":
    main()
