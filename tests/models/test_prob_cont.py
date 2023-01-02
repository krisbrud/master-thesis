#%%
import pytest
import torch
import torch.distributions

from models.auv_dreamer import AuvDreamerTorchPolicy

batch_size = 2500
horizon = 15

prob_continue_shape = (horizon, batch_size)
prob_continue_dist = torch.distributions.Uniform(
    torch.zeros(prob_continue_shape), 
    torch.ones(prob_continue_shape)
)
prob_continue_pred = prob_continue_dist.sample()

# TODO:
# Find out if the discount values should be used when calculating the value targets
# Answer:   

def sample_bernoulli_tensor(batch_size: int, p: float = 0.99):
    """"""
    return torch.bernoulli(torch.ones(batch_size) * p)

not_done = sample_bernoulli_tensor(2500).float().reshape(1, -1)

# TODO: Calculate cumulative product along horizon axis
discount_rates = torch.cat((not_done, prob_continue_pred[:-1]))

cumulative_prob_continue = torch.cumprod(discount_rates, 0)

# Shift the discount rates - as they measure whether the following state
# will be valid, not if the current state is valid

# The first element before shifting is the actual value of whether
# the time step from the replay buffer is done or not. However, we do not
# train the policy on the action taken leading up to the state from the replay
# buffer. This means that even though this may 

assert cumulative_prob_continue[2, 0] == (
      discount_rates[0, 0]
    * discount_rates[1, 0]
    * discount_rates[2, 0]
)

# discount_weights = , cumulative_prob_continue[:-1]))
# %%
