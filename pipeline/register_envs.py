from typing import List
from gym_auv import SCENARIOS


def _get_all_scenario_names() -> List[str]:
    scenario_names = list(SCENARIOS.keys())
    return scenario_names


def _register_one_env() -> None:
    # Registers one env in rllib registry
    pass


def register_gym_auv_scenarios() -> None:
    # Registers the available `gym-auv` scenarios as environments in the RLlib Tune framework
    pass
