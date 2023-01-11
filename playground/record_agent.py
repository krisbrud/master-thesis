# %%
# from argparse import ArgumentParser
import datetime
import numpy as np
import torch
import os
import gym
import gym.wrappers.monitoring.video_recorder as video_recorder
import gym_auv
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor
from ray.rllib.algorithms import Algorithm

from gym_auv.envs.movingobstacles import MovingObstaclesNoRules
from gym_auv.envs.testscenario import *
from torch import cuda

from models.auv_dreamer import AuvDreamer
from models.auv_dreamer_model import AuvDreamerModel
from pipeline.config import get_auv_dreamer_config_dict, get_ray_tune_auv_dreamer_config
from pipeline.register_components import register_gym_auv_scenarios

# def parse_arguments() -> dict:
#     parser = ArgumentParser()
#     # parser.add_argument("--model-checkpoint", type=str, help="Checkpoint of agent to use.")
#     args = parser.parse_args()
#     return args



register_gym_auv_scenarios()
env_name = "MovingObstaclesLosRewarder-v0"
gym_auv_config = gym_auv.Config()
# gym_auv_config.sensor.use_velocity_observations = False
dreamer_config = get_ray_tune_auv_dreamer_config(env_name, gym_auv.LOS_COLAV_CONFIG)
dreamer_config["prefill_timesteps"] = 0

device = "cpu"
if cuda.is_available(): # Use GPU if available
    dreamer_config["num_gpus"] = 1
    device = "cuda"


def filename_friendly_datetime() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat().replace(":","")
 

def make_recorder(env, name="testvideo"):
    this_directory = os.path.dirname(
        os.path.abspath(__file__)
    )  # Directory of this file
    video_directory = os.path.join(this_directory, "videos")

   
    if not os.path.exists(video_directory):
        os.mkdir(video_directory)

    video_path = os.path.join(video_directory, name)
    recorder = video_recorder.VideoRecorder(env=env, base_path=video_path)
    return recorder

def make_filename_datetime_suffix():
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")

env = gym.make(env_name)
# recorder = make_recorder(env, "checkpoint-760")
recorder = make_recorder(env,  "AuvDreamer-" + make_filename_datetime_suffix())

# algo = AuvDreamer(config=dreamer_config)  # .load_checkpoint(args.model_checkpoint)
# algo.load_checkpoint("/home/krisbrud/repos/master-thesis/playground/checkpoint-760")
# checkpoint_path = "/home/krisbrud/ray_results/AuvDreamer_2023-01-09_16-04-08/AuvDreamer_MovingObstaclesLosRewarder-v0_3fc10_00000_0_explore_noise=0.2000_2023-01-09_16-04-08/checkpoint_000001"
# checkpoint_path = "/home/krisbrud/ray_results/AuvDreamer_2023-01-10_12-41-06/AuvDreamer_MovingObstaclesLosRewarder-v0_0ce9d_00000_0_explore_noise=0.3000_2023-01-10_12-41-06/checkpoint_000020"
# checkpoint_path = "/home/krisbrud/ray_results/AuvDreamer_2023-01-10_12-41-06/AuvDreamer_MovingObstaclesLosRewarder-v0_0ce9d_00002_2_explore_noise=0.1000_2023-01-10_12-41-23/checkpoint_000040"
checkpoint_path = input("Checkpoint path:")
algo = Algorithm.from_checkpoint(checkpoint_path)
print("Loaded checkpoint!")

print("Starting recording rollout!")
state = []
observations = []
rewards = []
actions = []

obs = env.reset()
# while not done:
import tqdm
for i in tqdm.tqdm(range(1000)):
    # action = straight_ahead_action
    # obs = dict_flattening_preprocessor.transform(dict_obs)
    # obs = torch.Tensor(obs).view(1, -1).to(device)

    action, state, logp = algo.compute_single_action(obs, state, full_fetch=True) 

    # action = action.cpu().numpy().flatten()
    obs, reward, done, info = env.step(action)

    observations.append(obs)
    actions.append(action)
    rewards.append(reward)

    recorder.capture_frame()

    if done:
        print("Done inside record loop!", i)
        obs = env.reset()

recorder.close()

# %%
# Plot the rewards
import matplotlib.pyplot as plt
plt.figure()
plt.plot(rewards)
plt.title("rewards")
plt.figure()
plt.title("cumulative rewards")
plt.plot(np.cumsum(rewards))
# %%
policy = algo.get_policy()
model = policy.model

some_obs = observations[1]
some_action = actions[0]  # Action 0 leads to obs 1

some_action = torch.from_numpy(some_action).to(device).float().reshape((1, 1, -1))

obs_tensor = {
    k: torch.from_numpy(v).to(device).float() for (k, v) in some_obs.items()
    # "dense": torch.from_numpy(obs["dense"]).to(device).float(),
    # "lidar": torch.from_numpy(obs["lidar"]).to(device).float(),
}
obs_embedding = model.encoder.forward(obs_tensor).reshape((1, 1, -1))
# post, prior
post, prior = model.dynamics.observe(obs_embedding, some_action)
features = model.dynamics.get_feature(post)

some_obs_reconstruct = model.decoder(features)
some_obs_flat = torch.cat((obs_tensor["dense"], obs_tensor["lidar"].flatten()))
diff = some_obs_flat - some_obs_reconstruct.mean
# %%
