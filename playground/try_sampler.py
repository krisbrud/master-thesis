from gym import Env
from gym.spaces import Box
from typing import Tuple, Any

from ray.rllib.evaluation.sampler import SyncSampler
from ray.rllib.evaluation.rollout_worker import RolloutWorker

class MockEnv(Env):
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = Box(shape=(2,), low=-float("inf"), high=float("inf"))
        self.action_space = Box(shape=(1,), low=-float("inf"), high=float("inf"))
        self.counter = 0

    def step(self, action):
        self.counter += 1
        obs = [self.counter, action[0]]
        reward = 10 * self.counter + action[0]
        done = False
        info = {}

        return obs, reward, done, info 

    def reset(self):
        self.counter = 0
        return [self.counter, -1]

env_creator = lambda: MockEnv()
env = MockEnv()

worker = RolloutWorker(env_creator=env_creator)
sync_sampler = SyncSampler(worker=worker, env=env, clip_rewards=False, rollout_fragment_length=10)

sync_sampler.sample_collector