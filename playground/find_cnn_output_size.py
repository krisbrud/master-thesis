# %%
# Find the output size of the CNN encoder when using the occupancy grid

from ray.rllib.algorithms.dreamer.dreamer_model import ConvEncoder
import torch

occupancy_shape = (2, 64, 64)
cnn = ConvEncoder(shape=occupancy_shape)

mock_input = torch.zeros(occupancy_shape)
mock_output = cnn(mock_input)
# %%
