# Pytorch Models for using Dreamer together with `gym-auv`, based on Ray RLlib's implementation of the models for Dreamer
# https://github.com/ray-project/ray/blob/ea6d53dbf35a56bb87ecdfa2cc23bc9518a05f15/rllib/algorithms/dreamer/dreamer_model.py

# import torch
from typing import Any, Dict, Tuple, List
import torch
from torch import nn
from torch import distributions as td
from ray.rllib.utils.framework import TensorType

from ray.rllib.models.torch.misc import Reshape

# from ray.rllib.algorithms.dreamer.utils import Conv2d
from ray.rllib.algorithms.dreamer.utils import (
    Linear,
    Conv2d,
    ConvTranspose2d,
    GRUCell,
    TanhBijector,
)
from ray.rllib.algorithms.dreamer.dreamer_model import DenseDecoder, RSSM, ActionDecoder
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

from models.layers import Conv1d, ConvTranspose1d

ActFunc = Any

# Encoder, part of PlaNET
class AuvConvEncoder(nn.Module):
    """Standard Convolutional Encoder for Dreamer. This encoder is used
    to encode images frm an enviornment into a latent state for the
    RSSM model in PlaNET.
    """

    def __init__(
        self, depth: int = 32, act: ActFunc = None, shape: Tuple[int] = (3, 180)
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
        print(f"forward: {x.shape = }")
        orig_shape = list(x.size())

        # Last two dimensions are channels and bearings, the ones before may be
        # the batch size etc
        x = x.view(-1, *(orig_shape[-2:]))
        # x = x.view(-1, *(orig_shape[-3:]))
        x = self.model(x)

        new_shape = orig_shape[:-2] + [32 * self.depth]
        x = x.view(*new_shape)

        return x


class AuvEncoder(nn.Module):
    """Joint encoder for proprioceptive and lidar observations in gym_auv"""

    def __init__(self, navigation_shape=(6,), lidar_shape=(3, 180)):
        nav_hidden_size = 64
        self.navigation_encoder = nn.Sequential(
            [
                Linear(6, nav_hidden_size),
            ]
        )
        self.conv_encoder = AuvConvEncoder(shape=lidar_shape)

    def forward(self, x: Dict[str, TensorType]) -> TensorType:
        pass


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
            Reshape([-1, 32 * self.depth, 1]),
            ConvTranspose1d(32 * self.depth, 8 * self.depth, 5, stride=2),
            self.act(),
            ConvTranspose1d(8 * self.depth, 4 * self.depth, 5, stride=2),
            self.act(),
            ConvTranspose1d(4 * self.depth, 2 * self.depth, 6, stride=3),
            self.act(),
            ConvTranspose1d(2 * self.depth, self.depth, 6, stride=2),
            self.act(),
            ConvTranspose1d(self.depth, self.shape[0], 6, stride=2),
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


# Represents all models in Dreamer, unifies them all into a single interface
# Modified version of the original DreamerModel (https://github.com/ray-project/ray/blob/96cceb08e8bf73df990437002e25883c5a72d30c/rllib/algorithms/dreamer/dreamer_model.py),
# but adapted to be compatible with the input space of `gym-auv`
class AuvDreamerModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        nn.Module.__init__(self)
        self.depth = model_config["depth_size"]
        self.deter_size = model_config["deter_size"]
        self.stoch_size = model_config["stoch_size"]
        self.hidden_size = model_config["hidden_size"]

        self.action_size = action_space.shape[0]

        self.encoder = AuvConvEncoder(self.depth)
        self.decoder = AuvConvDecoder(
            self.stoch_size + self.deter_size, depth=self.depth
        )
        self.reward = DenseDecoder(
            self.stoch_size + self.deter_size, 1, 2, self.hidden_size
        )
        self.dynamics = RSSM(
            self.action_size,
            32 * self.depth,
            stoch=self.stoch_size,
            deter=self.deter_size,
        )
        self.actor = ActionDecoder(
            self.stoch_size + self.deter_size, self.action_size, 4, self.hidden_size
        )
        self.value = DenseDecoder(
            self.stoch_size + self.deter_size, 1, 3, self.hidden_size
        )
        self.state = None

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def policy(
        self, obs: TensorType, state: List[TensorType], explore=True
    ) -> Tuple[TensorType, List[float], List[TensorType]]:
        """Returns the action. Runs through the encoder, recurrent model,
        and policy to obtain action.
        """
        if state is None:
            self.state = self.get_initial_state(batch_size=obs.shape[0])
        else:
            self.state = state
        # TODO: Make clearer why this slicing is done
        post = self.state[:4]
        action = self.state[4]

        embed = self.encoder(obs)
        post, _ = self.dynamics.obs_step(post, action, embed)
        feat = self.dynamics.get_feature(post)

        action_dist = self.actor(feat)
        if explore:
            action = action_dist.sample()
        else:
            action = action_dist.mean
        logp = action_dist.log_prob(action)

        self.state = post + [action]
        return action, logp, self.state

    def imagine_ahead(self, state: List[TensorType], horizon: int) -> TensorType:
        """Given a batch of states, rolls out more state of length horizon."""
        start = []
        for s in state:
            s = s.contiguous().detach()
            shpe = [-1] + list(s.size())[2:]
            start.append(s.view(*shpe))

        def next_state(state):
            feature = self.dynamics.get_feature(state).detach()
            action = self.actor(feature).rsample()
            next_state = self.dynamics.img_step(state, action)
            return next_state

        last = start
        outputs = [[] for i in range(len(start))]
        for _ in range(horizon):
            last = next_state(last)
            [o.append(s) for s, o in zip(last, outputs)]
        outputs = [torch.stack(x, dim=0) for x in outputs]

        imag_feat = self.dynamics.get_feature(outputs)
        return imag_feat

    def get_initial_state(self) -> List[TensorType]:
        self.state = self.dynamics.get_initial_state(1) + [
            torch.zeros(1, self.action_space.shape[0]).to(self.device)
        ]
        # returned state should be of shape (state_dim, )
        self.state = [s.squeeze(0) for s in self.state]
        return self.state

    def value_function(self) -> TensorType:
        return None
