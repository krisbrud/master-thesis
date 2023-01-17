# %%
import numpy as np

# Use get_point_vector to get the vector at each point
# u, v = np.vectorize(get_point_vector)(x, y)

# Make a quiver plot in matplotlib
import matplotlib.pyplot as plt

# plt.quiver(x, y, u, v)
# plt.show()

# %%
from gym_auv.objects.path import Path

x_max = 400
n_points = 100
x_waypoints = np.linspace(0, x_max, n_points)

y_waypoints = np.cos((x_waypoints * np.pi / (x_max)) - (np.pi)) * (x_max) 

# plt.plot(x_waypoints, y_waypoints)

waypoints_np = np.array([x_waypoints, y_waypoints])

# Roatate waypoints by 30 degrees
# c, s = np.cos(theta), np.sin(theta)
# R = np.array(((c, -s), (s, c)))
# waypoints_np = R @ waypoints_np
print(waypoints_np.shape)
# waypoints = waypoints_np.tolist()
# print(waypoints)

waypoints_np = np.array([[0, 0], [80, 100], [300, 300]]).T

# Set axes to be equal
plt.axis("equal")
# Set xlim and ylim to 0, 300
plt.xlim(0, 300)
plt.ylim(-150, 150)

# plt.plot(waypoints_np[0, :], waypoints_np[1, :])
plt.show()
# Want waypoints in shape(2, N)
path = Path(waypoints_np)

# %%


def get_los_vector(x_coord, y_coord, path: Path = path, lookahead_distance=50):
    """Gets the Line-of-Sight vector at the point (x, y)"""
    pos = np.array([x_coord, y_coord])
    nearest_point_arclength = path.get_closest_arclength(pos)
    lookahead_arclength = nearest_point_arclength + lookahead_distance

    lookahead_point = path(lookahead_arclength)
    los_vector = lookahead_point - pos
    normalized_los_vector = los_vector / np.linalg.norm(los_vector)

    u_val, v_val = normalized_los_vector

    return u_val, v_val

# Make a meshgrid with spacing 50 between 0 and 300 along x and between -150 and 150 along y
x, y = np.meshgrid(np.linspace(0, 300, 7 + 6), np.linspace(0, 300, 7 + 6))


# Use get_los_vector to get the vector at each point
u, v = np.vectorize(get_los_vector)(x, y)

# Make a quiver plot in matplotlib
import matplotlib.pyplot as plt
# Set axes to be equal
plt.axis("equal")
# Set xlim and ylim to 0, 300
plt.xlim(0, 300)
plt.ylim(0, 300)
plt.quiver(x, y, u, v)
plt.plot(path.points[:,0], path.points[:,1])
# %%
import matplotlib.pyplot as plt



# plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Create figure and set x and y limits
fig, ax = plt.subplots()
ax.set_xlim([0, 300])
ax.set_ylim([0, 300])

# Set equal aspect ratio
ax.set_aspect('equal')

# Set titles
ax.set_title("Line-of-Sight Vector Field")

# Set labels
ax.set_xlabel(r"$x_n$")
ax.set_ylabel(r"$y_n$")

ax.quiver(x, y, u, v)
ax.plot(path.points[:,0], path.points[:,1])

# Show the figure
plt.show()

# %%
