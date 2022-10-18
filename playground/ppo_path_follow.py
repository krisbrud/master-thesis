import gym_auv
import gym
from pprint import pprint
from ray.rllib.algorithms.ppo import PPO
from pipeline.register_envs import register_gym_auv_scenarios
from ray.tune.registry import register_env
from gym_auv.envs.movingobstacles import PathFollowNoObstacles


def _env_factory(ignored_env_config):
    return PathFollowNoObstacles(gym_auv.PATHFOLLOW_CONFIG)


if __name__ == "__main__":
    register_gym_auv_scenarios()
    # env_name = "MovingObstaclesNoRules-v0"
    env_name = "PathFollowNoObstacles-v0"
    # PathFollowNoObstacles-v0

    # print("keys", gym_auv.SCENARIOS.keys())
    register_env(env_name, env_creator=_env_factory)
    ppo = PPO(env=env_name)

    for i in range(100):
        ppo.train()

    eval_results = ppo.evaluate()
    pprint(eval_results)
