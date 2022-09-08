from typing import List, Tuple
import torch
import pytest

from auv_dreamer_model import AuvConvDecoder, AuvConvEncoder
from gym_auv import Config
from ray.rllib.algorithms.dreamer import DreamerConfig
from ray.rllib.algorithms.dreamer.dreamer_model import ConvEncoder
from ray.rllib.algorithms.dreamer.utils import Linear
from ray.rllib.models.torch.misc import Reshape


from layers import Conv1d, ConvTranspose1d


def _get_lidar_shape() -> Tuple[int, int]:
    # Returns a tuple of the shape of the lidar measurements of a single timestep
    default_config = Config()
    lidar_shape = (3, default_config.vessel.n_sensors)

    return lidar_shape


def _get_rssm_feature_size() -> int:
    # Returns the feature size of the latent state, for use in testing the models
    dreamer_config = DreamerConfig()
    feature_size = (
        dreamer_config.dreamer_model["hidden_size"]
        + dreamer_config.dreamer_model["stoch_size"]
    )

    return feature_size


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


def test_reconstruction():
    # Test the encoder and decoder together, with sampling in latent space to avoid
    # "cheating" while reconstructing
    pass


def _print_shape(tensor: torch.Tensor):
    print(f"{tensor.shape = }")


def print_original_image_dimensions():
    original_shape_encoder = ConvEncoder()
    original_shape_image = torch.rand((1, 3, 64, 64))
    encoded_image = original_shape_encoder.forward(original_shape_image)
    print(f"{encoded_image.shape = }")  # = torch.Size([1, 1024])


def _apply_and_print_output_dim(x: torch.Tensor, layer: torch.nn.Module):
    out = layer(x)
    print(f"Applied layer {layer}.\n\tOutput shape: {out.shape}")
    return out


def _print_model_dims(x, layers: List[torch.nn.Module]):
    with torch.no_grad():
        out = x
        for layer in layers:
            out = _apply_and_print_output_dim(out, layer)


def experiment_with_encoder_conv_shapes():
    # pass

    lidar_shape = _get_lidar_shape()
    x = torch.rand(lidar_shape)
    x = x.view(-1, *(lidar_shape[-2:]))

    init_channels = lidar_shape[0]
    depth = 32
    kernel_size = 4

    print("input:")

    layers = [
        Conv1d(
            init_channels,
            depth,
            kernel_size,
            stride=2,
            dilation=1,
            padding_mode="circular",
            padding=2,
        ),
        Conv1d(depth, 2 * depth, kernel_size, stride=2),
        Conv1d(2 * depth, 4 * depth, kernel_size, stride=2),
        Conv1d(4 * depth, 4 * depth, kernel_size, stride=2),
    ]

    print("Encoder model dimensions:")
    _print_model_dims(x, layers)
    print("\n")


def experiment_with_decoder_conv_shapes():
    # pass

    lidar_shape = _get_lidar_shape()
    print(f"{lidar_shape = }")

    input_size = _get_rssm_feature_size()
    x = torch.rand(input_size)
    # x = x.view(-1, *(lidar_shape[-2:]))
    print(f"{x.shape = }")

    # init_channels = lidar_shape[0]
    depth = 32  # 32
    # kernel_size = 4

    layers = [
        Linear(input_size, 32 * depth),
        Reshape([-1, 32 * depth, 1]),
        # ConvTranspose1d(32 * depth, 4 * depth, 5, stride=2),
        # ConvTranspose1d(4 * depth, 2 * depth, 5, stride=2),
        # ConvTranspose1d(2 * depth, depth, 6, stride=2),
        # ConvTranspose1d(depth, lidar_shape[0], 6, stride=2),
        ConvTranspose1d(32 * depth, 8 * depth, 5, stride=2),
        ConvTranspose1d(8 * depth, 4 * depth, 5, stride=2),
        ConvTranspose1d(4 * depth, 2 * depth, 6, stride=3),
        ConvTranspose1d(2 * depth, depth, 6, stride=2),
        ConvTranspose1d(depth, lidar_shape[0], 6, stride=2),
    ]

    print("Decoder model dimensions:")
    _print_model_dims(x, layers)
    print("\n")


if __name__ == "__main__":
    # test_encoder()
    test_decoder(7)
    # test_reconstruction()
    # print_original_image_dimensions()
    # print_original_image_dimensions()
    # experiment_with_encoder_conv_shapes()
    # experiment_with_decoder_conv_shapes()
