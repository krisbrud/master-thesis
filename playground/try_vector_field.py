# %%

import numpy as np
x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))

u, v = x, y

some_other_val = 3

def get_point_vector(x_coord, y_coord, other_val=some_other_val):
    """Gets the vector at the point (x, y)"""
    u_val = np.cos(x_coord * other_val)
    v_val = np.sin(y_coord * other_val)

    return u_val, v_val

# Use get_point_vector to get the vector at each point
u, v = np.vectorize(get_point_vector)(x, y)

# Make a quiver plot in matplotlib
import matplotlib.pyplot as plt
plt.quiver(x, y, u, v)
plt.show()

# %%
