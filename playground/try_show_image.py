# %%
%load_ext autoreload
%autoreload 2
import gym
import gym_auv

from gym_auv.envs.testscenario import TestScenario1, TestScenario2
import copy
import numpy as np

# env_name = "TestScenario1-v0"
# env = gym.make(env_name)

# Run stable_baselines3 to test that the rendering works
config = copy.deepcopy(gym_auv.Config())
config.sensor.use_image_observation = True
# config.sensor.use

env = TestScenario2(
    gym_auv.DEFAULT_CONFIG, test_mode=False, verbose=True
)
obs = env.reset()
# %%
## Show image located in obs["image"]
import matplotlib.pyplot as plt
plt.imshow(obs["image"] / 255.0)
# %% 
plt.imshow(env.render() / 255.0)
# %%
