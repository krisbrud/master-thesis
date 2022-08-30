# Based on https://docs.ray.io/en/latest/rllib/index.html, but adapted to
# use custom gym environment

# Import custom environment - should automatically register the environment
# to the `gym` registry

import gym_auv
import gym
env_name = "TestScenario1-v0" 
env_config = gym_auv.SCENARIOS[env_name]['config']
env = gym.make(env_name, env_config=env_config)


# # Import the RL algorithm (Algorithm) we would like to use.
# from ray.rllib.algorithms.ppo import PPO

# # Configure the algorithm.
# config = {
#     # Environment (RLlib understands openAI gym registered strings).
#     "env": "Taxi-v3",
#     # Use 2 environment workers (aka "rollout workers") that parallelly
#     # collect samples from their own environment clone(s).
#     "num_workers": 2,
#     # Change this to "framework: torch", if you are using PyTorch.
#     # Also, use "framework: tf2" for tf2.x eager execution.
#     "framework": "tf",
#     # Tweak the default model provided automatically by RLlib,
#     # given the environment's observation- and action spaces.
#     "model": {
#         "fcnet_hiddens": [64, 64],
#         "fcnet_activation": "relu",
#     },
#     # Set up a separate evaluation worker set for the
#     # `algo.evaluate()` call after training (see below).
#     "evaluation_num_workers": 1,
#     # Only for evaluation runs, render the env.
#     "evaluation_config": {
#         "render_env": True,
#     },
# }

# # Create our RLlib Trainer.
# algo = PPO(config=config)

# # Run it for n training iterations. A training iteration includes
# # parallel sample collection by the environment workers as well as
# # loss calculation on the collected batch and a model update.
# for _ in range(3):
#     print(algo.train())

# # Evaluate the trained Trainer (and render each timestep to the shell's
# # output).
# algo.evaluate()
