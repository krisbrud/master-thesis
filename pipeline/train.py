import argparse

# from pprint import pprint
# from typing import Union
# from ray.tune.logger import DEFAULT_LOGGERS
import ray
from ray import tune
from ray import air
from ray.tune.search.bayesopt import BayesOptSearch

# import ray
# import datetime
from pipeline.register_envs import register_gym_auv_scenarios
from models.auv_dreamer import (
    AuvDreamer,
)
from pipeline import callbacks
from pipeline.config import get_ray_tune_auv_dreamer_config

# from ray.rllib.algorithms.dreamer.dreamer import DreamerConfig
from torch import cuda

# from ray.rllib.algorithms import Algorithm

import gym_auv


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
    )
    parser.add_argument(
        "--n-experiments",
        help="How many experiments to run in total. May typically run as many experiments as GPUs that are available",
        default=2,
    )
    # parser.add_argument("--device", type=Union[str, None], default=None)
    args = parser.parse_args()
    n_training_iters = args.train_iterations

    env_name = "MovingObstaclesNoRules-v0"
    # env_name = "PathFollowNoObstacles-v0"

    # gym_auv_config = gym_auv.Config()
    gym_auv_config = gym_auv.MOVING_CONFIG
    # auv_dreamer = auv_dreamer_factory(env_name)
    # dreamer_config = get_auv_dreamer_config_dict(
    #     env_name=env_name, gym_auv_config = gym_auv_config
    # )
    dreamer_config = get_ray_tune_auv_dreamer_config(
        env_name=env_name, gym_auv_config=gym_auv_config
    )
    # pprint(dreamer_config)
    # assert isinstance(dreamer_config, DreamerConfig)

    if cuda.is_available():
        # Use GPU if available
        dreamer_config["num_gpus"] = 1
    # dreamer_config.resources(num_gpus=1)


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
            num_samples=6,
        ),
        run_config=air.RunConfig(
            stop={"training_iteration": n_training_iters}, callbacks=[wandb_logger_callback]
        ),
        param_space=dreamer_config,
    )
    results = tuner.fit()
    print(results)

    # auv_dreamer = AuvDreamer(dreamer_config)
    # print("trying to save checkpoint!")uuu

    # mlflow.start_run(get_now_as_string())
    # for i in range(n_training_iters):
    #     print("training iteration", i)
    #     progress = auv_dreamer.train()
    #     pprint(progress)
    #     # print_last_reward(progress)

    #     if i % 100 == 0:
    #         auv_dreamer.save_checkpoint("results/")

    # print("evaluating!")
    # results = auv_dreamer.evaluate()
    # pprint(results)
    # mlflow.end_run()


if __name__ == "__main__":
    main()
