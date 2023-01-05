# %%
import numpy as np
import matplotlib.pyplot as plt

GRID_SIDE_SIZE = 64  # Height/width of grid
N_SENSORS = 180
SENSOR_RANGE = 150

sensor_angles = (np.arange(N_SENSORS) * 2 * np.pi / N_SENSORS) - np.pi
print(f"{sensor_angles[0] = }")
print(f"{sensor_angles[-1] = }")

ranges = np.linspace(0, SENSOR_RANGE, N_SENSORS)
collisions = np.array([True] * N_SENSORS)
print(f"{ranges = }")


pos = (
    np.vstack([ranges * np.cos(sensor_angles), ranges * np.sin(sensor_angles)])
).T  # Each row are (x, y) coordinates of a ray
pos_with_collisions = pos[collisions, :]
plt.scatter(pos_with_collisions[:, 0], pos_with_collisions[:, 1])
plt.show()

indices_decimals = (pos_with_collisions * (GRID_SIDE_SIZE / 2) / SENSOR_RANGE) + (GRID_SIDE_SIZE / 2)
indices = np.floor(indices_decimals).astype(np.int32)
# print(f"{indices = }")

# Occupancy grid uses (row, col) i.e. (y, x) indexing
occupancy_grid = np.zeros((GRID_SIDE_SIZE, GRID_SIDE_SIZE))
occupancy_grid[indices[:, 1], indices[:, 0]] = 1.0 
# plt.plot()
plt.imshow(occupancy_grid, origin="lower")
# plt.show()
print(f"{np.mean(occupancy_grid) = }")

# %%
