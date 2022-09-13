from typing import List, Tuple
import torch
import pytest
from models.auv_dreamer import _get_auv_dreamer_model_options

from models.auv_dreamer_model import AuvConvDecoder, AuvConvEncoder, AuvDreamerModel

import gym_auv
from gym_auv import Config

# from ray.rllib.algorithms.dreamer import DreamerConfig
# from ray.rllib.algorithms.dreamer.dreamer_model import ConvEncoder
# from ray.rllib.algorithms.dreamer.utils import Linear
# from ray.rllib.models.torch.misc import Reshape
import gym

from tests.models.model_utils import _get_lidar_shape, _get_rssm_feature_size


# TODO: Test encoder for different batch sizes
def test_encoder():
    # Make a encoder, check that it passes data of the correct dimensions
    # and supports backprop
    default_config = Config()
    lidar_shape = _get_lidar_shape()

    encoder = AuvConvEncoder(shape=lidar_shape)
    random_lidar_input = torch.rand(lidar_shape)

    latent_embedding = encoder.forward(random_lidar_input)
    print(f"{latent_embedding.shape = }")


@pytest.mark.parametrize("batch_size", [1, 7])
def test_decoder(batch_size):
    # Make a reconstructive decoder, check that it passes data of the correct dimensions
    # and supports backprop
    n_rssm_features = _get_rssm_feature_size()
    mock_latent_embedding = torch.rand((batch_size, n_rssm_features))

    lidar_shape = _get_lidar_shape()

    decoder = AuvConvDecoder(n_rssm_features, shape=lidar_shape)
    reconstruction_dist = decoder.forward(mock_latent_embedding)
    reconstruction = reconstruction_dist.sample()

    assert isinstance(reconstruction, torch.Tensor)

    expected_shape = (batch_size, *lidar_shape)
    assert reconstruction.shape == expected_shape, (
        "Reconstruction not of correct shape!"
        f"Expected shape {expected_shape} but got {reconstruction.shape}!"
    )


def _make_mock_auv_env() -> gym.Env:
    mock_auv_env = gym.make("TestScenario1-v0")
    return mock_auv_env


def test_auv_dreamer_model_initialization():
    # Make a mock auv env to get access to the observation and
    # action spaces
    mock_auv_env = _make_mock_auv_env()

    observation_space = mock_auv_env.observation_space
    action_space = mock_auv_env.action_space

    num_outputs = 1  # TODO find out value of this
    model_config = _get_auv_dreamer_model_options()
    name = "dreamer-test"

    auv_dreamer_model = AuvDreamerModel(
        obs_space=observation_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=model_config,
    )
