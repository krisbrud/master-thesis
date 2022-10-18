import argparse
from pprint import pprint
from typing import Union
from ray import tune
import datetime
from pipeline.register_envs import register_gym_auv_scenarios
from models.auv_dreamer import (
    AuvDreamer,
)
from pipeline.config import get_auv_dreamer_config_dict, get_ray_tune_auv_dreamer_config
from ray.rllib.algorithms.dreamer.dreamer import DreamerConfig
from torch import cuda
from ray.rllib.algorithms import Algorithm

import gym_auv

# import mlflow

# def train_iteration(algo: Algorithm, verbose=True):
#     progress = algo.train()
#     if verbose:
#         pprint(progress)

#     # mlflow.log_metric(key="progress", )


def main():
    # Register environments from gym_auv
    register_gym_auv_scenarios()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-iterations",
        help="How many training iterations (rollouts + learning) to perform during the course of training.",
        default=1000,
    )
    # parser.add_argument("--device", type=Union[str, None], default=None)
    args = parser.parse_args()
    n_training_iters = args.train_iterations

    # env_name = "MovingObstaclesNoRules-v0"
    env_name = "PathFollowNoObstacles-v0"

    gym_auv_config = gym_auv.Config()
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

    tuner = tune.Tuner(
        tune.with_resources(
            AuvDreamer,
            {"cpu": 8, "gpu": 3},
        ),
        tune_config=tune.TuneConfig(
            metric="episode_reward_mean",
            mode="max",
            # scheduler=
            num_samples=1,
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
