import argparse
from ray import tune

from pipeline.register_envs import register_gym_auv_scenarios
from models.auv_dreamer import AuvDreamer, auv_dreamer_factory, get_auv_dreamer_config

from ray.rllib.algorithms.dreamer.dreamer import DreamerConfig


def main():
    # Register environments from gym_auv
    register_gym_auv_scenarios()

    env_name = "MovingObstaclesNoRules-v0"
    # auv_dreamer = auv_dreamer_factory(env_name)
    dreamer_config = get_auv_dreamer_config()

    assert isinstance(dreamer_config, DreamerConfig)

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    if args.gpu:
        dreamer_config.resources(num_gpus=1)

    auv_dreamer = AuvDreamer(dreamer_config)

    for i in range(10):
        auv_dreamer.train()

    auv_dreamer.evaluate()


if __name__ == "__main__":
    main()
