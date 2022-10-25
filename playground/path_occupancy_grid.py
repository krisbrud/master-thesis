# %%
import numpy as np

import gym_auv
from gym_auv.envs.movingobstacles import MovingObstaclesNoRules
from gym_auv.objects.path import RandomCurveThroughOrigin
from gym_auv.objects.vessel.sensor import make_occupancy_grid
from gym_auv.utils.geomutils import Rz
from gym.utils import seeding
gym_auv_config = gym_auv.DEFAULT_CONFIG

nwaypoints = 4
seed = 321
rng, seed = seeding.np_random(seed)
path = RandomCurveThroughOrigin(rng, nwaypoints, length=800)

points = path.points[::10]  # Points are close, only need every 10 or so
# Filter out points that aren't close to the vessel
psi_heading = np.deg2rad(40)
position = np.array([0, 0])
init_state = np.hstack([position, psi_heading])
rotation_matrix = Rz(psi_heading)[:2, :2]
points_body = (points - init_state[:2]) @ rotation_matrix

path_occupancy_grid = make_occupancy_grid(positions_body=points_body, sensor_range=150, grid_size=64)

# %%
