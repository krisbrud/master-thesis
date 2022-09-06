from typing import Tuple
import torch

from auv_dreamer_model import AuvConvDecoder, AuvConvEncoder
from gym_auv import Config
from ray.rllib.algorithms.dreamer import DreamerConfig
from ray.rllib.algorithms.dreamer.dreamer_model import ConvEncoder


from layers import Conv1d


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


def test_encoder():
    # Make a encoder, check that it passes data of the correct dimensions
    # and supports backprop
    default_config = Config()
    lidar_shape = _get_lidar_shape()

    encoder = AuvConvEncoder(shape=lidar_shape)
    random_lidar_input = torch.rand(lidar_shape)

    latent_embedding = encoder.forward(random_lidar_input)
    print(f"{latent_embedding.shape = }")


def test_decoder():
    # Make a reconstructive decoder, check that it passes data of the correct dimensions
    # and supports backprop
    n_rssm_features = _get_rssm_feature_size()
    latent_embedding = torch.rand(n_rssm_features)

    lidar_shape = _get_lidar_shape()

    decoder = AuvConvDecoder(n_rssm_features, shape=lidar_shape)
    reconstruction = decoder.forward(latent_embedding)

    assert isinstance(reconstruction, torch.Tensor)

    assert reconstruction.shape == lidar_shape, (
        "Reconstruction not of correct shape!"
        f"Expected shape {lidar_shape} but got {reconstruction.shape}!"
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


def experiment_with_encoder_conv_shapes():
    # pass

    lidar_shape = _get_lidar_shape()
    x = torch.rand(lidar_shape)
    x = x.view(-1, *(lidar_shape[-2:]))

    init_channels = lidar_shape[0]
    depth = 32
    kernel_size = 4

    print("input:")
    _print_shape(x)
    conv1 = Conv1d(
        init_channels,
        depth,
        kernel_size,
        stride=2,
        dilation=1,
        padding_mode="circular",
        padding=2,
    )
    out1 = conv1.forward(x)
    print("after first")
    _print_shape(out1)

    conv2 = Conv1d(depth, 2 * depth, kernel_size, stride=2)
    out2 = conv2(out1)
    _print_shape(out2)

    out3 = Conv1d(2 * depth, 4 * depth, kernel_size, stride=2)(out2)
    # out3 = Conv1d(2 * depth, 2 * depth, kernel_size, stride=2)(out2)
    _print_shape(out3)

    out4 = Conv1d(4 * depth, 4 * depth, kernel_size, stride=2)(out3)
    _print_shape(out4)

    # out5 = Conv1d(4 * depth, 4 * depth, kernel_size, stride=2)(out4)
    # _print_shape(out5)


if __name__ == "__main__":
    # test_encoder()
    # test_decoder()
    # test_reconstruction()
    # print_original_image_dimensions()
    experiment_with_encoder_conv_shapes()
