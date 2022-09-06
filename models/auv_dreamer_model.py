# Pytorch Models for using Dreamer together with `gym-auv`, based on Ray RLlib's implementation of the models for Dreamer
# https://github.com/ray-project/ray/blob/ea6d53dbf35a56bb87ecdfa2cc23bc9518a05f15/rllib/algorithms/dreamer/dreamer_model.py

# import torch
from typing import Any, Tuple
from torch import nn
from torch import distributions as td

from ray.rllib.models.torch.misc import Reshape

# from ray.rllib.algorithms.dreamer.utils import Conv2d
from ray.rllib.algorithms.dreamer.utils import (
    Linear,
    Conv2d,
    ConvTranspose2d,
    GRUCell,
    TanhBijector,
)

from layers import Conv1d

ActFunc = Any

# Encoder, part of PlaNET
class AuvConvEncoder(nn.Module):
    """Standard Convolutional Encoder for Dreamer. This encoder is used
    to encode images frm an enviornment into a latent state for the
    RSSM model in PlaNET.
    """

    def __init__(
        self, depth: int = 32, act: ActFunc = None, shape: Tuple[int] = (3, 64, 64)
    ):
        """Initializes Conv Encoder

        Args:
            depth: Number of channels in the first conv layer
            act: Activation for Encoder, default ReLU
            shape: Shape of observation input
        """
        super().__init__()
        self.act = act
        if not act:
            self.act = nn.ReLU
        self.depth = depth
        self.shape = shape

        init_channels = self.shape[0]
        self.layers = [
            # Add circular padding in first layer, essentially keeping the circular physical
            # structure of the model, avoiding a "seam"
            Conv1d(init_channels, self.depth, 4, stride=2, padding_mode="circular"),
            self.act(),
            Conv1d(self.depth, 2 * self.depth, 4, stride=2),
            self.act(),
            Conv1d(2 * self.depth, 4 * self.depth, 4, stride=2),
            self.act(),
            Conv1d(4 * self.depth, 8 * self.depth, 4, stride=2),
            self.act(),
        ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        # TODO: Update
        # Flatten to [batch*horizon, 3, 64, 64] in loss function
        orig_shape = list(x.size())

        # Last two dimensions are channels and bearings, the ones before may be
        # the batch size etc
        x = x.view(-1, *(orig_shape[-2:]))
        # x = x.view(-1, *(orig_shape[-3:]))
        x = self.model(x)

        # new_shape = orig_shape[:-3] + [32 * self.depth]
        # x = x.view(*new_shape)
        return x


# Decoder, part of PlaNET
# class ConvDecoder(nn.Module):
class AuvConvDecoder(nn.Module):
    """Standard Convolutional Decoder for Dreamer.
    This decoder is used to decode images from the latent state generated
    by the transition dynamics model. This is used in calculating loss and
    logging gifs for imagined trajectories.
    """

    def __init__(
        self,
        input_size: int,
        depth: int = 32,
        act: ActFunc = None,
        shape: Tuple[int] = (3, 64, 64),
    ):
        """Initializes a ConvDecoder instance.
        Args:
            input_size: Input size, usually feature size output from
                RSSM.
            depth: Number of channels in the first conv layer
            act: Activation for Encoder, default ReLU
            shape: Shape of observation input
        """
        super().__init__()
        self.act = act
        if not act:
            self.act = nn.ReLU
        self.depth = depth
        self.shape = shape

        self.layers = [
            Linear(input_size, 32 * self.depth),
            Reshape([-1, 32 * self.depth, 1, 1]),
            ConvTranspose2d(32 * self.depth, 4 * self.depth, 5, stride=2),
            self.act(),
            ConvTranspose2d(4 * self.depth, 2 * self.depth, 5, stride=2),
            self.act(),
            ConvTranspose2d(2 * self.depth, self.depth, 6, stride=2),
            self.act(),
            ConvTranspose2d(self.depth, self.shape[0], 6, stride=2),
        ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        # x is [batch, hor_length, input_size]
        orig_shape = list(x.size())
        x = self.model(x)

        reshape_size = orig_shape[:-1] + list(self.shape)
        mean = x.view(*reshape_size)

        # Equivalent to making a multivariate diag
        return td.Independent(td.Normal(mean, 1), len(self.shape))
