import math
from typing import Dict, List, Tuple
import torch
import pytest
from models.auv_dreamer import _get_auv_dreamer_model_options

from models.auv_dreamer_model import (
    AuvConvDecoder1d,
    AuvConvEncoder1D,
    AuvDecoder,
    AuvDreamerModel,
    AuvEncoder,
)

import gym_auv
from gym_auv.envs.testscenario import TestScenario1
from gym_auv import Config

# from ray.rllib.algorithms.dreamer import DreamerConfig
# from ray.rllib.algorithms.dreamer.dreamer_model import DenseDecoder

# from ray.rllib.algorithms.dreamer.utils import Linear
# from ray.rllib.models.torch.misc import Reshape
import gym
from pipeline.config import get_auv_dreamer_config_dict

from tests.models.model_utils import (
    get_lidar_shape,
    get_rssm_feature_size,
    get_dense_size,
)

# TODO: Test encoder for different batch sizes
@pytest.mark.parametrize("batch_size", [1, 7])
def test_conv_encoder(batch_size):
    # Make a encoder, check that it passes data of the correct dimensions
    # and supports backprop
    default_config = Config()
    lidar_shape = get_lidar_shape()
    batched_shape = (batch_size, *lidar_shape)
    print(f"{batched_shape = }")

    encoder = AuvConvEncoder1D(shape=lidar_shape)
    random_lidar_input = torch.rand(batched_shape)

    latent_embedding = encoder.forward(random_lidar_input)
    # assert latent_embedding.shape == (1, 2), f"{latent_embedding.shape = }"
    print(f"{latent_embedding.shape = }")


def _make_mock_input(
    batch_size: int, n_proprio_states: int = 6, lidar_shape: Tuple[int, int] = (3, 180)
) -> Dict[str, torch.TensorType]:
    proprioceptive = torch.rand(
        (
            batch_size,
            6,
        )
    )
    lidar = torch.rand((batch_size, *lidar_shape))

    mock_input = {
        "proprioceptive": proprioceptive,
        "lidar": lidar,
    }
    return mock_input


def _mock_input(batch_size, input_size) -> torch.Tensor:
    return torch.rand((batch_size, input_size))

@pytest.fixture
def occupancy_grid_shape() -> Tuple[int,int,int]:
    return (2, 64, 64)

@pytest.mark.parametrize("batch_size", [1, 7])
def test_encoder(batch_size, occupancy_grid_shape):
    lidar_shape = get_lidar_shape()
    dense_size = get_dense_size()
    mock_env = TestScenario1(gym_auv.DEFAULT_CONFIG)
    obs_space = mock_env.observation_space
    encoder = AuvEncoder(dense_size=dense_size, 
        lidar_shape=lidar_shape, 
        occupancy_grid_shape=occupancy_grid_shape, 
        obs_space=obs_space,
        hidden_size=400,
        use_lidar=True,
        use_occupancy_grid=False,
        )

    # input_size = dense_size + math.prod(lidar_shape)
    # mock_input = _mock_input(batch_size=batch_size, input_size=input_size)
    # mock_input = _make_mock_input(
    #     batch_size=batch_size,
    #     n_proprio_states=navigation_shape[0],
    #     lidar_shape=lidar_shape,
    # )
    mock_input = {
        "dense": torch.rand((batch_size, dense_size)),
        "lidar": torch.rand((batch_size, *lidar_shape)),
    }
    embedding = encoder(mock_input)
    # assert embedding.shape == (1, 1024), f"{embedding.shape = }, not the expected!"


@pytest.mark.parametrize("batch_size", [1, 7])
def test_conv_decoder(batch_size):
    # Make a reconstructive decoder, check that it passes data of the correct dimensions
    # and supports backprop
    n_rssm_features = get_rssm_feature_size()
    mock_latent_embedding = torch.rand((batch_size, n_rssm_features))

    lidar_shape = get_lidar_shape()

    decoder = AuvConvDecoder1d(n_rssm_features, output_shape=lidar_shape)
    reconstruction = decoder.forward(mock_latent_embedding)

    assert isinstance(reconstruction, torch.Tensor)

    flat_shape = lidar_shape[0] * lidar_shape[1]
    expected_shape = (batch_size, flat_shape)
    assert reconstruction.shape == expected_shape, (
        "Reconstruction not of correct shape!"
        f"Expected shape {expected_shape} but got {reconstruction.shape}!"
    )


def _make_mock_auv_env() -> gym.Env:
    mock_auv_env = gym.make("TestScenario1-v0")
    return mock_auv_env


@pytest.fixture
def latent_size() -> int:
    return 230


@pytest.fixture
def hidden_size() -> int:
    return 1024


@pytest.fixture(name="latents", params=(1, 7))
def _latents(request, latent_size):
    batch_size = request.param
    latents = torch.rand((batch_size, latent_size))
    return latents


def test_decoder(latents, latent_size, occupancy_grid_shape):
    dense_size = get_dense_size()
    lidar_shape = get_lidar_shape()
    mock_scale = 1e-3
    decoder = AuvDecoder(latent_size, dense_size, lidar_shape, occupancy_grid_shape, mock_scale, mock_scale, use_lidar=True)
    out = decoder(latents)


def test_auv_dreamer_model_initialization():
    # Make a mock auv env to get access to the observation and
    # action spaces
    mock_auv_env = _make_mock_auv_env()

    observation_space = mock_auv_env.observation_space
    action_space = mock_auv_env.action_space

    num_outputs = 1  # TODO find out value of this
    # model_config = _get_auv_dreamer_model_options()
    env_name = "TestScenario1-v0"
    auv_dreamer_config = get_auv_dreamer_config_dict(
        env_name=env_name, gym_auv_config=gym_auv.DEFAULT_CONFIG
    )
    model_config = auv_dreamer_config["dreamer_model"]
    name = "dreamer-test"

    auv_dreamer_model = AuvDreamerModel(
        obs_space=observation_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=model_config,
        name=name,
    )


if __name__ == "__main__":
    test_encoder(1)
    test_decoder(1)
    # test_conv_encoder(1)
    # test_conv_decoder(1)
    # test_auv_dreamer_model_initialization()
