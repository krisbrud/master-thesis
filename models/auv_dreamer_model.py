# Pytorch Models for using Dreamer together with `gym-auv`, based on Ray RLlib's implementation of the models for Dreamer
# https://github.com/ray-project/ray/blob/ea6d53dbf35a56bb87ecdfa2cc23bc9518a05f15/rllib/algorithms/dreamer/dreamer_model.py

# import torch
import math
import copy
# from turtle import forward
import gym
import gym.spaces
from typing import Any, Dict, Tuple, List, Union
import torch
from torch import nn
import torch.functional as F
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
from ray.rllib.algorithms.dreamer.dreamer_model import (
    DenseDecoder,
    RSSM,
    ActionDecoder,
    ConvEncoder,
    ConvDecoder,
)
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions

from models.layers import Conv1d, ConvTranspose1d
from models.unflatten_obs import unflatten_obs

ActFunc = Any

# Encoder, part of PlaNET
class AuvConvEncoder1D(nn.Module):
    """Standard Convolutional Encoder for Dreamer. This encoder is used
    to encode images frm an enviornment into a latent state for the
    RSSM model in PlaNET.
    """

    def __init__(
        self,
        shape: Tuple[int],
        depth: int = 32,
        act: ActFunc = None,
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
            Conv1d(init_channels, self.depth * 2, 4, stride=2), #, padding_mode="circular"),
            self.act(),
            Conv1d(self.depth * 2, self.depth * 4, 4, stride=2),
            self.act(),
            Conv1d(self.depth * 4, self.depth * 8, 4, stride=2),
            self.act(),
            Conv1d(self.depth * 8, self.depth * 8, 4, stride=2),
            self.act(),
            Conv1d(self.depth * 8, self.depth * 8, 4, stride=2),
            self.act(),
            Conv1d(self.depth * 8, self.depth * 16, 4, stride=2),
            self.act(),
        ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        # TODO: Update
        # Flatten to [batch*horizon, 3, 180] in loss function
        orig_shape = list(x.size())

        # Last two dimensions are channels and bearings, the ones before may be
        # the batch size etc
        x = x.view(-1, *(orig_shape[-2:]))
        # x = x.view(-1, *(orig_shape[-3:]))
        x = self.model(x)

        # new_shape = orig_shape[:-2] + [32 * self.depth]
        single_output_shape = x.shape[-2] * x.shape[-1]
        new_shape = orig_shape[:-2] + [single_output_shape]

        x = x.view(*new_shape)

        return x


class AuvEncoder(nn.Module):
    """Joint encoder for proprioceptive and lidar observations in gym_auv"""

    def __init__(
        self,
        dense_size: int,
        lidar_shape: Tuple[int, int],
        occupancy_grid_shape: Tuple[int, int, int],
        obs_space: gym.spaces.Space,
        hidden_size: int,
        use_lidar: bool = True,
        use_occupancy_grid: bool = False,
    ):
        super().__init__()
        self.lidar_shape = lidar_shape
        self.dense_size = dense_size
        self.use_lidar = use_lidar
        self.use_occupancy_grid = use_occupancy_grid
        self.occupancy_grid_shape = occupancy_grid_shape
        # if self.use_lidar:
        #     self.flattened_size = self.dense_size + lidar_shape[0] * lidar_shape[1]
        # else:
        #     self.flattened_size = self.dense_size

        self.nav_hidden_size = hidden_size
        self.nav_output_size = 32 
        # self.hidden_output_size = 1024 + self.nav_output_size

        if self.use_lidar:
            if self.use_occupancy_grid:
                self.lidar_encoder = ConvEncoder(shape=self.occupancy_grid_shape)
                self.lidar_encoded_size = 1024  # Always depth**2 = 32 ** 2 = 1024
            else:
                self.lidar_encoder = AuvConvEncoder1D(shape=lidar_shape)
                self.lidar_encoded_size = (
                    1024
                )  # TODO: Refactor so this is given as argument
        else:
            self.lidar_encoder = None
            self.lidar_encoded_size = 0

        self.navigation_encoder = nn.Sequential(
            Linear(self.dense_size, self.nav_hidden_size),
            nn.ELU(),
            # nn.LayerNorm(self.nav_hidden_size),
            Linear(self.nav_hidden_size, self.nav_hidden_size),
            nn.ELU(),
            # nn.LayerNorm(self.nav_hidden_size),
            Linear(self.nav_hidden_size, self.nav_hidden_size),
            nn.ELU(),
            # nn.LayerNorm(self.nav_hidden_size),
            Linear(self.nav_hidden_size, self.nav_hidden_size),
            nn.ELU(),
            # nn.LayerNorm(self.nav_hidden_size),
            Linear(self.nav_hidden_size, self.nav_output_size),
            # Linear(self.dense_size, self.nav_output_size)
        )
        lidar_flat_size = math.prod(self.lidar_shape)
        self.output_size = self.nav_output_size + self.lidar_encoded_size # + self.dense_size

        # self.joint_head = nn.Sequential(
        #     # Linear(
        #     #     self.nav_output_size + self.lidar_encoded_size, self.hidden_output_size
        #     # )
        #     Linear(
        #         self.dense_size + lidar_flat_size, self.hidden_output_size 
        #     )
        # )

    def forward(self, x: Dict[str, TensorType]) -> TensorType:
        if self.use_lidar:
            # nav_obs, lidar_obs = unflatten_obs(
            #     x, lidar_shape=self.lidar_shape, dense_size=self.dense_size
            # )
            nav_obs = x["dense"]
            if self.use_occupancy_grid:
                lidar_obs = x["occupancy"]
            else:
                lidar_obs = x["lidar"]

            nav_latents = self.navigation_encoder(nav_obs)

            lidar_latents = self.lidar_encoder(lidar_obs)

            # if lidar_latents.isnan().any():
            #     print("Lidar latents contains NaNs")
            #     breakpoint()

            # Concatenate the encoded lidar and navigation latents, as well as the
            # raw navigation observations
            latents = torch.cat((nav_latents, lidar_latents), dim=-1)
            
            # Squeeze in order to remove 1-dim from (1, n_lidars)
            # Concat along observation dim
            # latents = torch.cat((nav_obs, lidar_obs.squeeze(-2)), dim=1) 
            # return raw_outputs
        else:
            latents = self.navigation_encoder(x)
            # latents = nav_obs

        # out = self.joint_head(latents)

        # return out
        return latents


# Decoder, part of PlaNET
# class ConvDecoder(nn.Module):
class AuvConvDecoder1d(nn.Module):
    """Standard Convolutional Decoder for Dreamer.
    This decoder is used to decode images from the latent state generated
    by the transition dynamics model. This is used in calculating loss and
    logging gifs for imagined trajectories.
    """

    def __init__(
        self,
        input_size: int,
        output_shape: Tuple[int, int],
        depth: int = 32,
        act: ActFunc = None,
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
        self.output_shape = output_shape

        self.layers = [
            Linear(input_size, 32 * self.depth),
            Reshape([-1, 32 * self.depth, 1]),
            ConvTranspose1d(32 * self.depth, 8 * self.depth, 5, stride=2),
            self.act(),
            ConvTranspose1d(8 * self.depth, 4 * self.depth, 5, stride=2),
            self.act(),
            ConvTranspose1d(4 * self.depth, 4 * self.depth, 5, stride=2),
            self.act(),
            ConvTranspose1d(4 * self.depth, 2 * self.depth, 5, stride=2),
            self.act(),
            ConvTranspose1d(2 * self.depth, self.depth, 6, stride=2),
            self.act(),
            ConvTranspose1d(
                self.depth, self.output_shape[0], 6, stride=2
            ),  # TODO: Get number of channels automatically
        ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        # x is [batch, hor_length, input_size]
        orig_shape = list(x.size())
        x = self.model(x)

        # reshape_size = orig_shape[:-1] + list(self.shape)
        # mean = x.view(*reshape_size)
        mean = x.view((*orig_shape[:-1], -1))  # Make sample dimension flat

        # Equivalent to making a multivariate diag
        # return td.Independent(td.Normal(mean, 1), len(self.shape))
        return mean



class AuvDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        dense_size: int,
        lidar_shape: Tuple[int, int],
        occupancy_grid_shape: Tuple[int, int, int],
        dense_decoder_scale: float,
        lidar_decoder_scale: float,
        use_lidar: bool = True,
        use_occupancy_grid: bool = False,
    ) -> None:
        super().__init__()

        self.lidar_shape = lidar_shape
        self.use_lidar = use_lidar
        self.dense_size = dense_size
        self.input_size = input_size
        self.dense_decoder_scale = dense_decoder_scale
        self.lidar_decoder_scale = lidar_decoder_scale
        self.dense_hidden_size = 400
        self.n_dense_layers = 4

        self.use_occupancy_grid = use_occupancy_grid
        self.occupancy_grid_shape = occupancy_grid_shape
        self.output_size = self.dense_size
        if self.use_lidar:
            if self.use_occupancy_grid:
                self.output_size += math.prod(self.occupancy_grid_shape)
                self.lidar_decoder = ConvDecoder(input_size=input_size, shape=self.occupancy_grid_shape)  # type: ignore
            else:
                self.output_size += math.prod(self.lidar_shape)
                # self.lidar_decoder = nn.Sequential(
                #     Linear(input_size, 64),
                #     nn.ReLU(),
                #     Linear(64, 180)
                # )
                self.lidar_decoder = AuvConvDecoder1d(
                    input_size, output_shape=lidar_shape
                )
        else:
            self.lidar_decoder = None

        navigation_decoder_layers = [
            Linear(self.input_size, self.dense_hidden_size),
        ]
        for i in range(self.n_dense_layers):
            navigation_decoder_layers.extend([
                Linear(self.dense_hidden_size, self.dense_hidden_size),
                nn.LayerNorm(self.dense_hidden_size),
                nn.ELU(),
            ])
        navigation_decoder_layers.append(Linear(self.dense_hidden_size, dense_size))
        self.navigation_decoder = nn.Sequential(*navigation_decoder_layers)

    def forward(self, x: torch.TensorType):
        leading_shape = x.shape[:-1]
        navigation_reconstruction = self.navigation_decoder(x)

        if self.use_lidar:
            lidar_reconstruction = self.lidar_decoder(x)
            if isinstance(lidar_reconstruction, torch.distributions.Independent):
                # Take care of case where we use the ConvEncoder from RLlib as the occupancy decoder,
                # s.t. the mean of lidar_reconstruction needs to be extracted. 
                lidar_reconstruction = lidar_reconstruction.mean
            
            flat_lidar_reconstruction = lidar_reconstruction.view((*leading_shape, -1))
            
            raw_mean = torch.cat(
                (navigation_reconstruction, flat_lidar_reconstruction), dim=-1
            )
        else:
            raw_mean = navigation_reconstruction

        mean = raw_mean.view((*leading_shape, -1))

        scale = torch.ones(self.output_size).to(x.device)
        scale[: self.dense_size] = self.dense_decoder_scale
        scale[self.dense_size :] = self.lidar_decoder_scale
        output_dist = td.Independent(td.Normal(mean, scale), 1)
        return output_dist

# class DiscountDecoder(nn.Module):
#     def __init__(self, input_size: int, hidden_size: int):
#         self.input_size = input_size
#         self.hidden_size = hidden_size
        
#         self.layers = nn.Sequential(
#             Linear(input_size, hidden_size),
#             nn.ReLU(),
#             Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             Linear(hidden_size, 1)
#         )
#         self.dist =

#     def forward(self, x: torch.TensorType) -> torch.TensorType:
        
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size=400, output_size=400) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            Linear(input_size, hidden_size),
            nn.ReLU(),
            Linear(hidden_size, hidden_size),
            nn.ReLU(),
            Linear(hidden_size, hidden_size),
            nn.ReLU(),
            Linear(hidden_size, output_size)
        )
    
    def forward(self, x: torch.TensorType) -> torch.TensorType:
        # print("x.shape", x.shape)
        return self.layers(x)

class AuvRSSM(RSSM):
    """Override the RSSM class to be able to reset the state if the state is the first one"""
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)

    def observe(
            self, embed: TensorType, action: TensorType, state: List[TensorType] = None, is_firsts: TensorType = None
        ) -> Tuple[List[TensorType], List[TensorType]]:
        """Returns the corresponding states from the embedding from ConvEncoder
        and actions. This is accomplished by rolling out the RNN from the
        starting state through each index of embed and action, saving all
        intermediate states between.

        Args:
            embed: ConvEncoder embedding
            action: Actions
            state (List[TensorType]): Initial state before rollout

        Returns:
            Posterior states and prior states (both List[TensorType])
        """
        batch_size = action.size()[0]
        if state is None:
            # state = self.get_initial_state(batch_size)
            state = self.get_initial_state(batch_size)
            print("Reset state at start of observe! Batch size: ", batch_size)
            breakpoint()

        if embed.dim() <= 2:
            embed = torch.unsqueeze(embed, 1)

        if action.dim() <= 2:
            action = torch.unsqueeze(action, 1)

        embed = embed.permute(1, 0, 2)
        action = action.permute(1, 0, 2)

        priors = [[] for i in range(len(state))]
        posts = [[] for i in range(len(state))]
        last = (state, state)
        for index in range(len(action)):
            # Tuple of post and prior
            # if is_firsts is not None:
            #     if is_firsts[index]:
            #         breakpoint()
            #         # Reset the state
            #         print("Resetting state", index)
            #         initial_state = self.get_initial_state()
            #         last = (initial_state, initial_state)

            last = self.obs_step(last[0], action[index], embed[index])
            [o.append(s) for s, o in zip(last[0], posts)]
            [o.append(s) for s, o in zip(last[1], priors)]

        prior = [torch.stack(x, dim=0) for x in priors]
        post = [torch.stack(x, dim=0) for x in posts]

        prior = [e.permute(1, 0, 2) for e in prior]
        post = [e.permute(1, 0, 2) for e in post]

        breakpoint()

        return post, prior

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

        self.dense_size = model_config["dense_size"]
        self.lidar_shape = model_config["lidar_shape"]
        self.use_lidar = model_config["use_lidar"]
        self.use_occupancy_grid = model_config["use_occupancy"]
        self.occupancy_grid_shape = model_config["occupancy_grid_shape"]

        self.dense_decoder_scale = model_config[
            "dense_decoder_scale"
        ]  # Fixed scale parameter of gaussian in dense decoder
        self.lidar_decoder_scale = model_config[
            "lidar_decoder_scale"
        ]  # Same, but for lidar

        self.action_size = action_space.shape[0]

        self.encoder = AuvEncoder(
            self.dense_size,
            self.lidar_shape,
            self.occupancy_grid_shape,
            hidden_size=self.hidden_size,
            obs_space=self.obs_space,
            use_lidar=self.use_lidar,
            use_occupancy_grid=self.use_occupancy_grid,
        )
        # embed_size = 400
        # obs_size = obs_space.shape[0]
        # self.encoder = MLP(obs_size, self.hidden_size, embed_size)

        self.decoder = AuvDecoder(
            self.stoch_size + self.deter_size,
            self.dense_size,
            self.lidar_shape,
            self.occupancy_grid_shape,
            dense_decoder_scale=self.dense_decoder_scale,
            lidar_decoder_scale=self.lidar_decoder_scale,
            use_lidar=self.use_lidar,
            use_occupancy_grid=self.use_occupancy_grid,
        )
        # print("WARNING: Hard coded lidar shape in AuvDreamerModel")
        # self.decoder = DenseDecoder(self.stoch_size + self.deter_size, self.dense_size + 256, 3, self.hidden_size)
        
        self.reward = DenseDecoder(
            self.stoch_size + self.deter_size, 1, 3, self.hidden_size
        )

        embed_size = self.encoder.output_size
        # self.dynamics = RSSM(
        #     self.action_size,
        #     embed_size,
        #     # 32 * self.depth,
        #     # 1024 + 32,
        #     stoch=self.stoch_size,
        #     deter=self.deter_size,
        # )
        self.dynamics = AuvRSSM(
            self.action_size,
            embed_size,
            # 32 * self.depth,
            # 1024 + 32,
            stoch=self.stoch_size,
            deter=self.deter_size,
        )
        self.actor = ActionDecoder(
            self.stoch_size + self.deter_size,
            self.action_size,
            4,
            self.hidden_size,
            # mean_scale=2.0, # Default: 5
            act=nn.ELU,
        )
        self.value = DenseDecoder(
            self.stoch_size + self.deter_size, 1, 3, self.hidden_size
        )

        self.value_target = copy.deepcopy(self.value) 

        self.discount = DenseDecoder(
            self.stoch_size + self.deter_size, 1, 4, self.hidden_size, dist="binary"
        )

        self.state = None

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print("initialized auv dreamer model")

    def policy(
        self, obs: TensorType, state: List[TensorType], explore=True
    ) -> Tuple[TensorType, List[float], List[TensorType]]:
        """Returns the action. Runs through the encoder, recurrent model,
        and policy to obtain action.
        """
        obs_dict = restore_original_dimensions(
            obs=obs, obs_space=self.obs_space, tensorlib="torch"
        )
        # obs_dict = obs
        if state is None:
            self.state = self.get_initial_state() # batch_size=obs.shape[0])
        else:
            self.state = state
        # TODO: Make clearer why this slicing is done
        post = self.state[:4]
        action = self.state[4]

        embed = self.encoder(obs_dict)
        # breakpoint()
        # try:
        post, _ = self.dynamics.obs_step(post, action, embed)
        # except:
            # embed has nans

            # breakpoint()
        feat = self.dynamics.get_feature(post)

        action_dist = self.actor(feat)
        if explore:
            action = action_dist.sample()
        else:
            action = action_dist.mean
        logp = action_dist.log_prob(action)

        self.state = post + [action]
        return action, logp, self.state

    def imagine_ahead(self, state: List[TensorType], horizon: int, return_actions=False) -> TensorType:
        """Given a batch of states, rolls out more state of length horizon.
        
        return_actions (bool): If True, returns the actions taken as well. 
        This may be used for calculating entropy etc.
        """
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
        
    
        def next_state_return_action(state):
            feature = self.dynamics.get_feature(state).detach()
            action = self.actor(feature).rsample()
            next_state = self.dynamics.img_step(state, action)
            return next_state, action


        last = start
        outputs = [[] for i in range(len(start))]
        actions = []
        for _ in range(horizon):
            if return_actions:
                last, action = next_state_return_action(last)
                actions.append(action)
            else:
                last = next_state(last)
            
            last = next_state(last)
            [o.append(s) for s, o in zip(last, outputs)]
        outputs = [torch.stack(x, dim=0) for x in outputs]

        # breakpoint()

        imag_feat = self.dynamics.get_feature(outputs)
        
        if return_actions:
            actions = torch.stack(actions, dim=0)
            return imag_feat, actions
        else:
            return imag_feat

    def get_initial_state(self) -> List[TensorType]:
        self.state = self.dynamics.get_initial_state(1) + [
            torch.zeros(1, self.action_space.shape[0]).to(self.device)
        ]
        # returned state should be of shape (state_dim, )
        self.state = [s.squeeze(0) for s in self.state]
        return self.state

    def update_target_critic(self):
        """Copies the critic parameters to the target critic."""
        self.value_target.load_state_dict(self.value.state_dict())

    def value_function(self) -> TensorType:
        return None
