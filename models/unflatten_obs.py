from typing import Tuple
from torch import TensorType


def unflatten_obs(
    flat_obs, dense_size=(6,), lidar_shape=(3, 180)
) -> Tuple[TensorType, TensorType]:
    # Assumes observation of size [B, N],
    # Where B is batch size and N is the number of individual observations

    n_nav_obs = dense_size[0]
    nav_obs = flat_obs[:, :n_nav_obs]
    lidar_obs = flat_obs[:, n_nav_obs:].reshape(-1, *lidar_shape)

    return nav_obs, lidar_obs
