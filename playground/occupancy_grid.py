import numpy as np

GRID_SIDE_SIZE = 64  # Height/width of grid
N_SENSORS = 180
SENSOR_RANGE = 150


sensor_angles = (np.arange(N_SENSORS) * 2 * np.pi) - np.pi
print(f"{sensor_angles[0] = }")
print(f"{sensor_angles[-1] = }")

ranges = np.linspace(0, SENSOR_RANGE, N_SENSORS)
print(f"{ranges = }")

occupancy_grid = np.array((GRID_SIDE_SIZE, GRID_SIDE_SIZE))
# x_pos = ranges * np.cos(sensor_angles) 
# y_pos = ranges * np.sin(sensor_angles)

pos = np.vstack([np.cos(sensor_angles), np.sin(sensor_angles)]).T * ranges  # Each row are coordinates of a ray
print(f"{pos = }")

indices = pos // np.array([GRID_SIDE_SIZE / 2, GRID_SIDE_SIZE/ 2])
print(f"{indices = }")

occupancy_grid[indices] = 1.0

print(occupancy_grid)