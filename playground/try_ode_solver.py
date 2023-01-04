#%%
import numpy as np

# Import the ODE solver from scipy
from scipy.integrate import odeint

# Lorenz attractor
def dynamics(state, t):
    x, y, z = state
    return [10 * (y - x), x * (28 - z) - y, x * y - (8 / 3) * z]


# Initial conditions
state0 = [1.0, 1.0, 1.0]

# Time points
t = np.linspace(0, 100, 10000)

# Solve ODE
# Time the solver

import time
ping = time.time()
states = odeint(dynamics, state0, t)
pong = time.time()

print("Time elapsed: ", pong - ping, "s")

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
# Now solve the same problem using the Otter Vessel class
