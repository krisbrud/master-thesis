from ray.rllib.algorithms.dreamer import Dreamer, DreamerConfig

from models.auv_dreamer import auv_dreamer_factory
from models.auv_dreamer import _get_auv_dreamer_model_options
from pipeline.register_envs import register_gym_auv_scenarios


def test_auv_dreamer_initialization():
    # Register the gym environments in gym_auv so rllib can use them
    register_gym_auv_scenarios()

    # Make a Dreamer instance
    dreamer = auv_dreamer_factory("TestScenario1-v0", num_envs_per_worker=2)


if __name__ == "__main__":
    test_auv_dreamer_initialization()
