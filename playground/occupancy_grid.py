import numpy as np

GRID_SIDE_SIZE = 64  # Height/width of grid
N_SENSORS = 180
SENSOR_RANGE = 150

sensor_angles = (np.arange(N_SENSORS) * 2 / np.pi) - np.pi
print(f"{sensor_angles[0] = }")
print(f"{sensor_angles[-1] = }")

ranges = np.linspace(0, SENSOR_RANGE, N_SENSORS)
print(f"{ranges = }")

occupancy_grid = np.zeros((GRID_SIDE_SIZE, GRID_SIDE_SIZE))

pos = (
    np.vstack([ranges * np.cos(sensor_angles), ranges * np.sin(sensor_angles)])
).T  # Each row are coordinates of a ray
print(f"{pos = }")

collisions = np.array([True] * N_SENSORS)

indices_decimals = (pos * (GRID_SIDE_SIZE / 2) / SENSOR_RANGE) + (GRID_SIDE_SIZE / 2)
indices = np.floor(indices_decimals).astype(np.int32)
indices_with_collisions = indices[collisions, :]
print(f"{indices = }")

occupancy_grid[indices[:, 0], indices[:, 1]] = 1.0

print(f"{np.mean(occupancy_grid) = }")
