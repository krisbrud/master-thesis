from typing import List
import gym
from ray.tune import register_env

from gym_auv import SCENARIOS


def _get_all_scenario_names() -> List[str]:
    scenario_names = list(SCENARIOS.keys())
    return scenario_names


def _register_gym_env_to_rllib(env_name: str) -> None:
    # Registers one env in rllib registry given that it is registered in
    register_env(env_name, lambda ignored_env_config: gym.make(env_name))


def register_gym_auv_scenarios() -> None:
    # Registers the available `gym-auv` scenarios as environments in the RLlib Tune framework
    scenarios = _get_all_scenario_names()

    for scenario in scenarios:
        _register_gym_env_to_rllib(scenario)
