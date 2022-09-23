# Based on https://docs.ray.io/en/latest/rllib/index.html, but adapted to
# use custom gym environment

# Import custom environment - should automatically register the environment
# to the `gym` registry
# import torch
import gym_auv
import gym

# Import the RL algorithm (Algorithm) we would like to use.
from ray.rllib.algorithms.ppo import PPO
from ray import tune

from gym_auv.envs.testscenario import TestScenario1


def train():
    env_name = "TestScenario1-v0"
    env_config = gym_auv.SCENARIOS[env_name]["config"]
    # Register environment in RLlib
    tune.register_env(env_name, lambda env_config: TestScenario1(env_config))

    # Configure the algorithm.
    config = {
        # Environment (RLlib understands openAI gym registered strings).
        "env": env_name,  # "Taxi-v3",
        "env_config": {"config": env_config},
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "num_workers": 1,  # 2,
        "num_gpus": 0,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",  # tf",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },
        "train_batch_size": 200,
        "sgd_minibatch_size": 200,
        # Set up a separate evaluation worker set for the
        # `algo.evaluate()` call after training (see below).
        # "evaluation_duration_unit": "timesteps",
        "evaluation_num_workers": 1,
        "evaluation_duration": 1,
        "horizon": 200,  # env_config["max_timesteps"],
        # "soft_horizon": 200,
        # Only for evaluation runs, render the env.
        # "evaluation_config": {
        #     "render_mode": "2d",
        # },
    }

    # Create our RLlib Trainer.
    algo = PPO(config=config)

    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    n_training_iters = 1
    for i in range(n_training_iters):
        print(algo.train())
        print("Training iteration:", i)
        # algo.save_checkpoint("outputs")

    # print("Going to evaluate!")
    algo.evaluate()
    # print("Evaluation done!")


if __name__ == "__main__":
    train()

    # env.reset()
    # obs = env.step([0.5, 0.5])
    # print(obs)
