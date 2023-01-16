#%%
import dreamerv2.api as dv2

import gym
import gym_auv
import dreamerv2.api as dv2
from dreamerv2.api import common

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
import collections
import re
import pathlib

def prepare_agent(env, config, outputs = None):
    """Collect a random episode and use it to initialize the agent.
    
    This is basically an adaption of the train function from dreamerv2.api, 
    as the agent needs an episode to be initialized, otherwise loading a checkpoint
    may cause problems.
    """
    logdir = pathlib.Path(config.logdir).expanduser()
    # logdir.mkdir(parents=True, exist_ok=True)
    # config.save(logdir / 'config.yaml')
    print(config, '\n')
    print('Logdir', logdir)

    outputs = outputs or [
        common.TerminalOutput(),
        # common.JSONLOutput(config.logdir),
        # common.TensorBoardOutput(config.logdir),
    ]
    replay = common.Replay(logdir / 'train_episodes', **config.replay)
    step = common.Counter(replay.stats['total_steps'])
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = common.Every(config.train_every)
    should_log = common.Every(config.log_every)
    should_video = common.Every(config.log_every)
    should_expl = common.Until(config.expl_until)

    def per_episode(ep):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'Episode has {length} steps and return {score:.1f}.')
        logger.scalar('return', score)
        logger.scalar('length', length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f'sum_{key}', ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f'mean_{key}', ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f'max_{key}', ep[key].max(0).mean())
            if should_video(step):
                for key in config.log_keys_video:
                    logger.video(f'policy_{key}', ep[key])
        logger.add(replay.stats)
        logger.write()

    env = common.GymWrapper(env)
    env = common.ResizeImage(env)
    if hasattr(env.act_space['action'], 'n'):
        env = common.OneHotAction(env)
    else:
        env = common.NormalizeAction(env)
    env = common.TimeLimit(env, config.time_limit)

    driver = common.Driver([env])
    driver.on_episode(per_episode)
    driver.on_step(lambda tran, worker: step.increment())
    driver.on_step(replay.add_step)
    driver.on_reset(replay.add_step)

    prefill = max(0, config.prefill - replay.stats['total_steps'])
    if prefill:
        print(f'Prefill dataset ({prefill} steps).')
        random_agent = common.RandomAgent(env.act_space)
        driver(random_agent, steps=prefill, episodes=1)
        driver.reset()

    print('Create agent.')
    agnt = dv2.agent.Agent(config, env.obs_space, env.act_space, step)
    dataset = iter(replay.dataset(**config.dataset))
    train_agent = common.CarryOverState(agnt.train)
    train_agent(next(dataset))
    if (logdir / 'variables.pkl').exists():
        print("Loading agent from file")
        agnt.load(logdir / 'variables.pkl')
        return agnt
    # else:
    #     print('Pretrain agent.')
    #     for _ in range(config.pretrain):
    #     train_agent(next(dataset))
    # policy = lambda *args: agnt.policy(
    #     *args, mode='explore' if should_expl(step) else 'train')


logdir = "/home/krisbrud/repos/master-thesis/logdir/MovingObstaclesLosRewarder-v0-jan-14"
config = dv2.common.Config().load(logdir + "/config.yaml")

gym_auv_config = gym_auv.MOVING_CONFIG
gym_auv_config.sensor.use_image_observation = True

env_name = "MovingObstaclesLosRewarder-v0"
env = gym.make(env_name, env_config=gym_auv_config)

agnt = prepare_agent(env, config)
print(agnt)
# env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)

# env = dv2.common.GymWrapper(env)
# env = dv2.common.ResizeImage(env)
# if hasattr(env.act_space['action'], 'n'):
#     env = dv2.common.OneHotAction(env)
# else:
#     env = dv2.common.NormalizeAction(env)
# env = dv2.common.TimeLimit(env, config.time_limit)
# # env = CustomRGBImgPartialObsWrapper(env)

# obs_space = env.obs_space
# act_space = env.act_space

# agnt = dv2.agent.Agent(config, obs_space, act_space, 0)



# agnt.save("/home/krisbrud/repos/master-thesis/logdir/MovingObstaclesLosRewarder-v0-jan-14/foo-variables.pkl")
# try:
#     agnt.load(logdir + "/variables.pkl")
# except Exception as e:
#     print("e", e)
# dv2.train(env, config)
# %%
