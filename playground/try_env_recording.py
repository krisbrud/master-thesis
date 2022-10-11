import os
import gym
import gym.wrappers.monitoring.video_recorder as video_recorder
import gym_auv

from gym_auv.envs.testscenario import TestScenario1
from gym_auv.envs.movingobstacles import MovingObstaclesNoRules, PathFollowNoObstacles

import numpy as np

renderer = "2d"
# env = TestScenario1(
#     gym_auv.DEFAULT_CONFIG, test_mode=False, renderer=renderer, verbose=True
# )
# env = MovingObstaclesNoRules(
#     gym_auv.DEFAULT_CONFIG, test_mode=False, renderer=renderer, verbose=True
# )
env = PathFollowNoObstacles(
    gym_auv.PATHFOLLOW_CONFIG, test_mode=False, renderer=renderer, verbose=True
)


this_directory = os.path.dirname(os.path.abspath(__file__))  # Directory of this file
video_directory = os.path.join(this_directory, "videos")

if not os.path.exists(video_directory):
    os.mkdir(video_directory)

video_path = os.path.join(video_directory, "pathfollow")
recorder = video_recorder.VideoRecorder(env=env, base_path=video_path)

straight_ahead_action = np.array([0.9, 0.0])

obs = env.reset()
print("Evaluating!")
for i in range(1000):
    action = straight_ahead_action

    obs, reward, done, info = env.step(action)
    recorder.capture_frame()

    if done:
        obs = env.reset()

recorder.close()
