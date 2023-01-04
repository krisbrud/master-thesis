# %%
import gym_auv
from gym_auv.objects.vessel.otter import Otter3DoF

otter = Otter3DoF()

import numpy as np
from scipy.integrate import odeint

# Initial conditions
init_state = np.zeros((6,))

# Time points
t = np.linspace(0, 1, 10)

# Solve ODE
# Time the solver
circle_action = np.array([50, 0])
dynamics = lambda state, t: otter.dynamics(state[:3], state[3:], circle_action)

import time

durations = []

for i in range(10):
    ping = time.time()
    states = odeint(dynamics, init_state, t)
    pong = time.time()

    duration = pong - ping
    durations.append(duration)
    
    print("Time elapsed: ", pong - ping, "s")

print("Mean time elapsed: ", np.mean(durations), "s")
print("Std time elapsed: ", np.std(durations), "s")

# Plot the solution in 3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.plot(states[:, 0], states[:, 1], states[:, 2])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()
# %%
