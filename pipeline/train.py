from ray import tune

from pipeline.register_envs import register_gym_auv_scenarios
from models.auv_dreamer import auv_dreamer_factory, get_auv_dreamer_config


def main():
    # Register environments from gym_auv
    register_gym_auv_scenarios()

    env_name = "MovingObstaclesNoRules-v0"
    auv_dreamer = auv_dreamer_factory(env_name)

    for i in range(10):
        auv_dreamer.train()

    auv_dreamer.evaluate()


if __name__ == "__main__":
    main()
