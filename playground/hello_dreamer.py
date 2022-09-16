# Based on https://docs.ray.io/en/latest/rllib/index.html, but adapted to
# use custom gym environment

# Import custom environment - should automatically register the environment
# to the `gym` registry

import dataclasses
import gym_auv
import gym

# Import the RL algorithm (Algorithm) we would like to use.
from ray.rllib.algorithms.dreamer import Dreamer, DreamerConfig
from ray import tune

from gym_auv.envs.testscenario import TestScenario1


def train():
    env_name = "Taxi-v3"  # "TestScenario1-v0"
    # env_config = gym_auv.SCENARIOS[env_name]["config"]
    # print(f"{env_config = }")
    # env = gym.make(env_name, env_config=env_config)

    # Register environment in RLlib
    # tune.register_env(env_name, lambda env_config: TestScenario1(env_config))
    tune.register_env("Taxi-v3", lambda env_config: gym.make("Taxi-v3"))

    # Configure the algorithm.
    config = {
        # Environment (RLlib understands openAI gym registered strings).
        "env": env_name,  # "Taxi-v3",
        # "env_config": {"config": env_config},
        # Use 2 environment workers (aka "rollout workers") that parallelly
        # collect samples from their own environment clone(s).
        "num_workers": 1,  # 2,
        # Change this to "framework: torch", if you are using PyTorch.
        # Also, use "framework: tf2" for tf2.x eager execution.
        "framework": "torch",
        # Tweak the default model provided automatically by RLlib,
        # given the environment's observation- and action spaces.
        "model": {
            "fcnet_hiddens": [64, 64],
            "fcnet_activation": "relu",
        },
        "train_batch_size": 64,
        # Set up a separate evaluation worker set for the
        # `algo.evaluate()` call after training (see below).
        "evaluation_num_workers": 1,
        "horizon": 10000,  # env_config["max_timesteps"],
        # "soft_horizon": 200,
        # Only for evaluation runs, render the env.
        "evaluation_config": {
            "render_env": True,
        },
    }

    config = DreamerConfig()
    config.framework("torch")
    # config.env_config = {"config": env_config}

    # Create our RLlib Trainer.
    # algo = PPO(config=config)
    algo = Dreamer(env=env_name, config=config)  # , config=config)

    # Run it for n training iterations. A training iteration includes
    # parallel sample collection by the environment workers as well as
    # loss calculation on the collected batch and a model update.
    n_training_iters = 2
    for _ in range(n_training_iters):
        print(algo.train())
        algo.save_checkpoint("outputs")

    algo.evaluate()


# Evaluate the trained Trainer (and render each timestep to the shell's
# output).
# algo.evaluate()

if __name__ == "__main__":
    train()

    # env.reset()
    # obs = env.step([0.5, 0.5])
    # print(obs)