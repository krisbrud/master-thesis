# %%
import os
import gym
import gym.wrappers.monitoring.video_recorder as video_recorder
import gym_auv
import matplotlib.pyplot as plt
import tqdm
from gym_auv.envs.testscenario import TestScenario1
from gym_auv.envs.movingobstacles import MovingObstaclesNoRules, PathFollowNoObstacles

import numpy as np

renderer = "2d"
# env = TestScenario1(
#     gym_auv.DEFAULT_CONFIG, test_mode=False, renderer=renderer, verbose=True
# )
gym_auv_config = gym_auv.Config()
gym_auv_config.sensor.use_lidar = True
gym_auv_config.sensor.use_occupancy_grid = True

env = MovingObstaclesNoRules(
    gym_auv_config, test_mode=False, renderer=renderer, verbose=True
)
# env = PathFollowNoObstacles(
#     gym_auv.PATHFOLLOW_CONFIG, test_mode=False, renderer=renderer, verbose=True
# )


this_directory = os.path.dirname(os.path.abspath(__file__))  # Directory of this file
video_directory = os.path.join(this_directory, "videos")

if not os.path.exists(video_directory):
    os.mkdir(video_directory)

def make_filename_datetime_suffix():
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")

video_path = os.path.join(video_directory, "otter_pathfollow-" + make_filename_datetime_suffix())
recorder = video_recorder.VideoRecorder(env=env, base_path=video_path)

# straight_ahead_action = np.array([0.9, np.random.normal(0, 0.01)])
# straight_ahead_action = np.array([20, 15]) # 20])

observations = []
rewards = []
obs = env.reset()
print("Evaluating!")
for i in tqdm.tqdm(range(500)):    
    # action = straight_ahead_action
    # action = np.random.uniform(low=0, high=20, size=(2,))
    action = [0.8, 0.002]

    obs, reward, done, info = env.step(action)
    # print(i, action, env.vessel._state)
    observations.append(obs)
    rewards.append(reward)
    recorder.capture_frame()

    # print(i)

    if done:
        obs = env.reset()
recorder.close()
# %%
import matplotlib.pyplot as plt
# Plot rewards, clipped to [-2, 2]
clipped_rewards = np.clip(rewards, -100, 2)

plt.plot(clipped_rewards)
# %%
