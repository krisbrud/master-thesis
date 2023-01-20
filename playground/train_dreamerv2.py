import dreamerv2.api as dv2

import gym
import gym_auv
import dreamerv2.api as dv2
# import dv2pipeline.custom_dv2_api as dv2

import copy

log_suffix = "-jan-20-10"
# Override some of the default config options located in dreamerv2/configs.yaml
config = dv2.defaults.update({
    ## Training options
    "logdir": "~/repos/master-thesis/logdir/MovingObstaclesLosRewarder-v0" + log_suffix,
    "log_every": 1e3,
    "train_every": 10,
    "eval_every": 10e3, # Doesn't really do anything but write to tensorboard
    "prefill": 50e3, #5e4,
    # "envs": 4, 

    ## Agent options
    "clip_rewards": "tanh",
    "expl_noise": 0.05,  # Mainly use entropy instead of noise to encourage exploration
    "actor_ent": 1e-4,
    
    ## Model options
    "loss_scales.kl": 1.0,
    "discount": 0.995,  # 0.99 as default
    # "kl": {"free": 1.0}, 
    "kl": {"free": 1.0}, 
    "model_opt": {"lr": 1e-5},
    "actor_opt": {"lr": 1e-5},
    "critic_opt": {"lr": 1e-5},
    "rssm": {"deter": 200, "hidden": 200}, 
    "encoder": {"mlp_keys": "dense", "cnn_keys": "image"},
    "decoder": {"mlp_keys": "dense", "cnn_keys": "image"},
}).parse_flags()

gym_auv_config = gym_auv.LOS_COLAV_CONFIG
gym_auv_config.sensor.use_image_observation = True
gym_auv_config.episode.use_terminated_truncated_step_api = True
# gym_auv_config.sensor.n_lidar_rays = 90

env_name = "MovingObstaclesLosRewarder-v0"
env = gym.make(env_name, env_config=gym_auv_config)
# env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
print(env.config.sensor.n_lidar_rays)

# env = CustomRGBImgPartialObsWrapper(env)
# import tensorflow as tf
# tf.config.run_functions_eagerly(True)

# print(env)
dv2.train(env, config)