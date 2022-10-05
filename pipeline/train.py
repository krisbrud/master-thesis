from ray import tune

from pipeline.register_envs import register_gym_auv_scenarios
from models.auv_dreamer import AuvDreamer, auv_dreamer_factory, get_auv_dreamer_config

from ray.rllib.algorithms.dreamer.dreamer import DreamerConfig, Dreamer
from torch.cuda import is_available

import gym_auv

def make_env_config() -> dict:
    config = gym_auv.MOVING_CONFIG

    config.vessel.sensor_use_velocity_observations = False
    

def main():
    # Register environments from gym_auv
    register_gym_auv_scenarios()

    env_name = "MovingObstaclesNoRules-v0"

    # auv_dreamer = auv_dreamer_factory(env_name)
    dreamer_config = get_auv_dreamer_config(env_name=env_name)

    assert isinstance(dreamer_config, DreamerConfig)

    if is_available():
        # Use GPU if available
        dreamer_config.resources(num_gpus=1)

    # Decrease batch size (and learning rates)
    dreamer_config.batch_size = 10
    dreamer_config.td_model_lr /= 5
    dreamer_config.actor_lr /= 5
    dreamer_config.critic_lr /= 5
    dreamer_config.horizon = gym_auv.DEFAULT_CONFIG.episode.max_timesteps

    auv_dreamer = Dreamer(dreamer_config)
    # print("trying to save checkpoint!")
    # for i in range(10):
    #     print("training iteration", i)
    #     progress = auv_dreamer.train()
    #     print("progress", progress)

    #     if i % 5 == 0:
    #         auv_dreamer.save_checkpoint("results/")

    print("evaluating!")
    results = auv_dreamer.evaluate()
    print("results:\n", results)


if __name__ == "__main__":
    main()
