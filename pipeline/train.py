from pprint import pprint
from ray import tune

from pipeline.register_envs import register_gym_auv_scenarios
from models.auv_dreamer import (
    AuvDreamer,
)
from models.config import get_auv_dreamer_config_dict
from ray.rllib.algorithms.dreamer.dreamer import DreamerConfig
from torch import cuda
from ray.rllib.algorithms import Algorithm

import gym_auv


def train_iteration(algo: Algorithm, verbose=True):
    progress = algo.train()
    if verbose:
        pprint(progress)


def main():
    # Register environments from gym_auv
    register_gym_auv_scenarios()

    env_name = "MovingObstaclesNoRules-v0"
    gym_auv_config = gym_auv.Config()
    # auv_dreamer = auv_dreamer_factory(env_name)
    dreamer_config = get_auv_dreamer_config_dict(
        env_name=env_name, gym_auv_config = gym_auv_config
    )
    pprint(dreamer_config)
    # assert isinstance(dreamer_config, DreamerConfig)

    if cuda.is_available():
        # Use GPU if available
        dreamer_config["num_gpus"] = 1
        # dreamer_config.resources(num_gpus=1)

    auv_dreamer = AuvDreamer(dreamer_config)
    # print("trying to save checkpoint!")
    n_training_iters = 1
    for i in range(n_training_iters):
        print("training iteration", i)
        progress = auv_dreamer.train()
        print("progress", progress)

        if i % 5 == 0:
            auv_dreamer.save_checkpoint("results/")

    print("evaluating!")
    results = auv_dreamer.evaluate()
    pprint(results)


if __name__ == "__main__":
    main()
