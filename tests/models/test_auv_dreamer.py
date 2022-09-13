from ray.rllib.algorithms.dreamer import Dreamer, DreamerConfig

from models.auv_dreamer import auv_dreamer_factory
from models.auv_dreamer import _get_auv_dreamer_model_options


def test_auv_dreamer_initialization():
    dreamer = auv_dreamer_factory("TestScenario1-v0")
