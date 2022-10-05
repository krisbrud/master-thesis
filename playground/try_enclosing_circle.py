import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import gym_auv.objects.obstacles as obstacles


def plot_points_and_enclosing_circle(points: np.ndarray):
    rect = obstacles.PolygonObstacle(points)
    circle = obstacles.enclosing_circle_of_shape(rect._calculate_boundary())
    np_points = np.array(points)
    circle_patch = patches.Circle(circle.center, circle.radius, fill=False)
    plt.gca().add_patch(circle_patch)
    plt.plot(np_points[:, 0], np_points[:, 1])
    plt.show()


points1 = [(0, 0), (0, 1), (0.5, 1), (1, 1), (1, 0)]
points2 = [(0, -1), (-1, 0), (2, 1), (1, 2), (0, -1)]

plot_points_and_enclosing_circle(points1)
plot_points_and_enclosing_circle(points2)