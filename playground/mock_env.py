import numpy as np
import gym
from gym.spaces.box import Box


class MockEnv(gym.Env):
    def __init__(self, ignored_envconfig):
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,))
        self.observation_space = Box(low=-1.0, high=1.0, shape=(1,))
        self.reset()

    def reset(self):
        self.counter = 0
        obs = self._get_observation()

        return obs

    def step(self, action):
        self.counter += 1

        obs = self._get_observation()
        reward = 0.1
        info = {}

        done = False
        if self.counter == 123:
            # Give a done signal at a random time
            done = True

        return obs, reward, done, info

    def _get_observation(self):
        return np.array([0.5])
