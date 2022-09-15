import gym
import gym_auv

from gym_auv.envs.testscenario import TestScenario1

from stable_baselines3 import PPO
import numpy as np

# env_name = "TestScenario1-v0"
# env = gym.make(env_name)

# Run stable_baselines3 to test that the rendering works
render_mode = None  # "2d"
env = TestScenario1(
    gym_auv.DEFAULT_CONFIG, test_mode=False, render_mode=render_mode, verbose=True
)

straight_ahead_action = np.array([0.9, 0.0])

model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=500)

obs = env.reset()
print("Evaluating!")
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    # action = straight_ahead_action
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
