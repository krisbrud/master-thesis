#%%
import dreamerv2.api as dv2

import gym
import gym_auv
import dreamerv2.api as dv2

import copy

# log_suffix = "-jan-14"
# Override some of the default config options located in dreamerv2/configs.yaml
# config = dv2.defaults.update({
#     ## Training options
#     "logdir": "~/repos/master-thesis/logdir/MovingObstaclesLosRewarder-v0" + log_suffix,
#     "log_every": 1e3,
#     "train_every": 10,
#     "eval_every": 1e4,
#     "prefill": 2e4,
#     # "envs": 4, 

#     ## Agent options
#     "clip_rewards": "tanh",
#     "expl_noise": 0.0,  # Use entropy instead of noise to encourage exploration
#     "actor_ent": 1e-4,
    
#     ## Model options
#     "loss_scales.kl": 1.0,
#     "discount": 0.99,
#     "kl": {"free": 1.0}, 
#     "model_opt": {"lr": 3e-4},
#     "actor_opt": {"lr": 8e-5},
#     "critic_opt": {"lr": 8e-5},
#     "rssm": {"deter": 200, "hidden": 200}, 
#     "encoder": {"mlp_keys": "dense", "cnn_keys": "image"},
#     "decoder": {"mlp_keys": "dense", "cnn_keys": "image"},
# }).parse_flags()


def prepare_agent(config, env):
    """Collect a random episode and use it to initialize the agent."""


logdir = "/home/krisbrud/repos/master-thesis/logdir/MovingObstaclesLosRewarder-v0-jan-14"
config = dv2.common.Config().load(logdir + "/config.yaml")

gym_auv_config = gym_auv.MOVING_CONFIG
gym_auv_config.sensor.use_image_observation = True

env_name = "MovingObstaclesLosRewarder-v0"
env = gym.make(env_name, env_config=gym_auv_config)
# env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)

env = dv2.common.GymWrapper(env)
env = dv2.common.ResizeImage(env)
if hasattr(env.act_space['action'], 'n'):
    env = dv2.common.OneHotAction(env)
else:
    env = dv2.common.NormalizeAction(env)
env = dv2.common.TimeLimit(env, config.time_limit)
# env = CustomRGBImgPartialObsWrapper(env)

obs_space = env.obs_space
act_space = env.act_space

agnt = dv2.agent.Agent(config, obs_space, act_space, 0)



# agnt.save("/home/krisbrud/repos/master-thesis/logdir/MovingObstaclesLosRewarder-v0-jan-14/foo-variables.pkl")
try:
    agnt.load(logdir + "/variables.pkl")
except Exception as e:
    print("e", e)
# dv2.train(env, config)
# %%
