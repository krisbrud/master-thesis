import argparse

import ray
from ray import tune
from ray import air
from ray.tune.search.bayesopt import BayesOptSearch
from torch import cuda

import gym_auv

from pipeline import callbacks
from pipeline.config import get_ray_tune_auv_dreamer_config
from pipeline.register_envs import register_gym_auv_scenarios
from models.auv_dreamer import (
    AuvDreamer,
)


def main():
    ray.init()
    print(ray.available_resources())
    # Register environments from gym_auv
    register_gym_auv_scenarios()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-iterations",
        help="How many training iterations (rollouts + learning) to perform during the course of training.",
        default=100,
        type=int,
    )
    parser.add_argument(
        "--n-experiments",
        help="How many experiments to run in total. May typically run as many experiments as GPUs that are available",
        default=2,
        type=int,
    )
    args = parser.parse_args()
    n_training_iters = args.train_iterations

    env_name = "MovingObstaclesNoRules-v0"

    gym_auv_config = gym_auv.MOVING_CONFIG
    dreamer_config = get_ray_tune_auv_dreamer_config(
        env_name=env_name, gym_auv_config=gym_auv_config
    )

    if cuda.is_available():
        # Use GPU if available
        dreamer_config["num_gpus"] = 1

    # Populated from environment variables
    wandb_logger_callback = callbacks.get_wandb_logger_callback()

    tuner = tune.Tuner(
        tune.with_resources(
            AuvDreamer,
            {"cpu": 2, "gpu": 1},
        ),
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            num_samples=args.n_experiments,
        ),
        run_config=air.RunConfig(
            stop={"training_iteration": n_training_iters},
            callbacks=[wandb_logger_callback],
        ),
        param_space=dreamer_config,
    )
    results = tuner.fit()
    print(results
        modified:   playground/try_hierarchical_detector.py)


if __name__ == "__main__":
    main()
