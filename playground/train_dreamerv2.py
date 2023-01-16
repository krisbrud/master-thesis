import dreamerv2.api as dv2

import gym
import gym_auv
import dreamerv2.api as dv2

import copy

log_suffix = "-jan-16-1"
# Override some of the default config options located in dreamerv2/configs.yaml
config = dv2.defaults.update({
    ## Training options
    "logdir": "~/repos/master-thesis/logdir/MovingObstaclesLosRewarder-v0" + log_suffix,
    "log_every": 1e3,
    "train_every": 10,
    "eval_every": 1e4,
    "prefill": 2.5e4,
    # "envs": 4, 

    ## Agent options
    "clip_rewards": "tanh",
    "expl_noise": 0.0,  # Use entropy instead of noise to encourage exploration
    "actor_ent": 1e-4,
    
    ## Model options
    "loss_scales.kl": 1.0,
    "discount": 0.995,  # 0.99 as default
    "kl": {"free": 1.0}, 
    "model_opt": {"lr": 3e-4},
    "actor_opt": {"lr": 8e-5},
    "critic_opt": {"lr": 8e-5},
    "rssm": {"deter": 200, "hidden": 200}, 
    "encoder": {"mlp_keys": "dense", "cnn_keys": "image"},
    "decoder": {"mlp_keys": "dense", "cnn_keys": "image"},
}).parse_flags()

gym_auv_config = gym_auv.LOS_COLAV_CONFIG
gym_auv_config.sensor.use_image_observation = True

env_name = "MovingObstaclesLosRewarder-v0"
env = gym.make(env_name, env_config=gym_auv_config)
# env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)

# env = CustomRGBImgPartialObsWrapper(env)

# print(env)
dv2.train(env, config)