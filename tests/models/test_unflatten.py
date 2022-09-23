# Test unflattening of navigation and lidar observations

import pytest
import torch

from models.unflatten_obs import unflatten_obs


@pytest.fixture(name="flat_obs", params=(1, 7))
def _flat_obs(request) -> torch.TensorType:
    batch_size = request.param
    return torch.Tensor(list(range(546))).reshape(-1, 546)


@pytest.fixture
def nav_obs(flat_obs) -> torch.TensorType:
    nav_obs, _ = unflatten_obs(flat_obs)
    return nav_obs


@pytest.fixture
def lidar_obs(flat_obs) -> torch.TensorType:
    _, lidar_obs = unflatten_obs(flat_obs)
    return lidar_obs


def test_nav_shape(nav_obs: torch.TensorType):
    assert nav_obs.shape[-1] == 6


def test_lidar_shape(lidar_obs: torch.TensorType):
    assert lidar_obs.shape[-2:] == (3, 180)


def test_nav_values(nav_obs: torch.TensorType):
    assert nav_obs[0, 0] == 0


def test_lidar_first_value(lidar_obs: torch.TensorType):
    assert lidar_obs[0, 0, 0] == 6


def test_lidar_second_value(lidar_obs: torch.TensorType):
    assert lidar_obs[0, 0, 1] == 7


def test_lidar_second_channel_value(
    nav_obs: torch.TensorType, lidar_obs: torch.TensorType
):
    n_rays = lidar_obs.shape[-1]
    n_navigation_features = nav_obs.shape[-1]
    assert lidar_obs[0, 1, 0] == n_navigation_features + n_rays


def test_lidar_second_sample_value(lidar_obs: torch.TensorType):
    if lidar_obs.shape[0] > 1:
        n_features = lidar_obs.shape[-1]
        assert lidar_obs[1, 0, 0] == n_features + 6
