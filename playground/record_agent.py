from argparse import ArgumentParser
import datetime
import numpy as np
import torch
import os
import gym
import gym.wrappers.monitoring.video_recorder as video_recorder
import gym_auv
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor

from gym_auv.envs.movingobstacles import MovingObstaclesNoRules
from gym_auv.envs.testscenario import *
from torch import cuda

from models.auv_dreamer import AuvDreamer
from models.auv_dreamer_model import AuvDreamerModel
from pipeline.config import get_auv_dreamer_config_dict, get_ray_tune_auv_dreamer_config
from pipeline.register_envs import register_gym_auv_scenarios

def parse_arguments() -> dict:
    parser = ArgumentParser()
    # parser.add_argument("--model-checkpoint", type=str, help="Checkpoint of agent to use.")
    args = parser.parse_args()
    return args


# args = parse_arguments()
renderer = "2d"
env = MovingObstaclesNoRules(
    gym_auv.DEFAULT_CONFIG, test_mode=False, renderer=renderer, verbose=True
)

# env = TestScenario2(
#     gym_auv.DEFAULT_CONFIG, test_mode=False, renderer=renderer, verbose=True
# )

register_gym_auv_scenarios()
env_name = "MovingObstaclesNoRules-v0"
# env_name = "TestScenario2-v0"
gym_auv_config = gym_auv.Config()
# auv_dreamer = auv_dreamer_factory(env_name)
# dreamer_config = get_auv_dreamer_config_dict(
#     env_name=env_name, gym_auv_config=gym_auv_config
# )
dreamer_config = get_ray_tune_auv_dreamer_config(env_name, gym_auv.DEFAULT_CONFIG)
dreamer_config["prefill_timesteps"] = 0


# dreamer_config["dreamer_model"] = {
#             "use_lidar": False,
#             "dense_size": 6,
#             "depth_size": 32,
#             "deter_size": 48,
#             "stoch_size": 24,
#             "hidden_size": 478,
#             "lidar_shape": [
#                 1,
#                 180
#             ],
#             "custom_model": "",
#             "action_init_std": 5
# }

device = "cpu"
if cuda.is_available(): # Use GPU if available
    dreamer_config["num_gpus"] = 1
    device = "cuda"
    # dreamer_config.resources(num_gpus=1)

print("using device", device)

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

recorder = make_recorder(env, "checkpoint-760")
straight_ahead_action = np.array([0.9, 0.0])

algo = AuvDreamer(config=dreamer_config)  # .load_checkpoint(args.model_checkpoint)
algo.load_checkpoint("/home/krisbrud/repos/master-thesis/playground/checkpoint-760")
print("Loaded checkpoint!")

obs = env.reset()
done = False
model = algo.get_policy().model

assert isinstance(model, AuvDreamerModel) # Help language server with type hints
state = model.get_initial_state()
state = [s.view(1,-1).to(device) for s in state]

dict_flattening_preprocessor = DictFlatteningPreprocessor(obs_space=env.observation_space)

print("Starting recording rollout!")
# while not done:
for i in range(1000):
    # action = straight_ahead_action
    obs = torch.Tensor(obs).view(1, -1).to(device)
    action, _, state = model.policy(obs, state)

    action = action.cpu().numpy().flatten()
    dict_obs, reward, done, info = env.step(action)
    obs = dict_flattening_preprocessor.transform(dict_obs)
    recorder.capture_frame()

    if done:
        print("Done inside record loop!", i)
        obs = env.reset()

recorder.close()
 