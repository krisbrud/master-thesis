# %% 
import gym_auv
import gym

env = gym.make("MovingObstaclesLosRewarder-v0")

obs = env.reset()

action = [1, 0]

new_obs, reward, done, info = env.step(action)

# %%
