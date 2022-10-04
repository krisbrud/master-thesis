from turtle import position
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import patches
import shapely.geometry 
import gym_auv

from gym_auv.objects.obstacles import BaseObstacle, CircleParams, CircularObstacle
from gym_auv.objects.vessel.vessel import Vessel
from gym_auv.objects.vessel.sensor import _find_rays_to_simulate_for_obstacles


def make_vessel() -> Vessel:
    pos = [1, 2]
    heading = np.deg2rad(30)
    init_state = [*pos, heading]

    return Vessel(gym_auv.DEFAULT_CONFIG, init_state=init_state)


def make_obstacle() -> CircularObstacle:
    pos = np.array([4, 7])
    radius = 1

    return CircularObstacle(pos, radius)


def plot_situation(vessel: Vessel, obstacle: BaseObstacle):
    # Plot sensors
    n_rays = len(vessel._sensor_angles)
    sensor_origin = np.array([vessel.position for _ in range(n_rays)])
    sensor_angles_body = vessel._sensor_angles + vessel.heading

    heading = vessel.heading
    sensor_range = vessel.config.vessel.sensor_range
    sensor_ray_end = (
        np.array([np.sin(sensor_angles_body), np.cos(sensor_angles_body)]).T
        * sensor_range
        + sensor_origin
    )

    ray_x = np.array([sensor_origin[:, 1], sensor_ray_end[:, 1]])
    ray_y = np.array([sensor_origin[:, 0], sensor_ray_end[:, 0]])

    plt.plot(ray_x, ray_y, "r")

    obstacles_per_ray = _find_rays_to_simulate_for_obstacles(
        [obstacle],
        shapely.geometry.Point(*vessel.position),
        vessel.heading,
        vessel._d_sensor_angle,
        n_rays=vessel.config.vessel.n_sensors,
    )

    n_obstacles_per_ray = np.array([len(obsts) for obsts in obstacles_per_ray])
    idx_rays_with_obstacles = n_obstacles_per_ray > 0
    plt.plot(ray_x[:,idx_rays_with_obstacles], ray_y[:, idx_rays_with_obstacles], "b")
    heading_line_x = np.array([0, np.sin(heading)]) * sensor_range + vessel.position[1]
    heading_line_y = np.array([0, np.cos(heading)]) * sensor_range + vessel.position[0]
    plt.plot(heading_line_x, heading_line_y, "g")

    circle_patch = patches.Circle(
        np.array(obstacle.enclosing_circle.center)[::-1],
        obstacle.enclosing_circle.radius,
        fill=False,
    )
    plt.gca().add_patch(circle_patch)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.show()
    input("")


vessel = make_vessel()
obstacle = make_obstacle()

plot_situation(vessel, obstacle)
