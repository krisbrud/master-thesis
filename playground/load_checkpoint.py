# %%
from ray.rllib.algorithms.algorithm import Algorithm

# Take the checkpoint path as an argument using argparse
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--checkpoint_path", type=str, default="/home/krisbrud/ray_results/AuvDreamer_2023-01-09_16-04-08/AuvDreamer_MovingObstaclesLosRewarder-v0_3fc10_00000_0_explore_noise=0.2000_2023-01-09_16-04-08/checkpoint_0000014")
# args = parser.parse_args()

# checkpoint_path = args.checkpoint_path
checkpoint_path = "/home/krisbrud/ray_results/AuvDreamer_2023-01-09_16-04-08/AuvDreamer_MovingObstaclesLosRewarder-v0_3fc10_00000_0_explore_noise=0.2000_2023-01-09_16-04-08/checkpoint_000001"

loaded_checkpoint = Algorithm.from_checkpoint(checkpoint=checkpoint_path)
print("jada")

import gym
import gym_auv

env = gym.make("MovingObstaclesLosRewarder-v0")

algo = loaded_checkpoint

obs = env.reset()

actions = algo.compute_single_action(obs, full_fetch=False)

print(actions)  # Returns actions, state, logp
# %%
