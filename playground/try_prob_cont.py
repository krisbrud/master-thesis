#%%
import torch

gamma = 0.99
foo = torch.ones(2500, 15)

bar = foo.cumprod(dim=-1)
# Print bar
print(bar)
# %%

def set_first_element_to_actual(prob_cont: torch.Tensor, actual: torch.Tensor):
    """Sets the first element of x in every batch to the actual value"""
    prob_cont[:, 0] = actual
    return prob_cont

# Make a function that makes a tensor of shape (batch_size,)
# that contains ones with probability p=0.99 per element and zero otherwise
def sample_bernoulli_tensor(batch_size: int, p: float = 0.99):
    """"""
    return torch.bernoulli(torch.ones(batch_size) * p)

rollout_started = sample_bernoulli_tensor(2500)

baz = set_first_element_to_actual(foo, rollout_started) * gamma



# %%

# %%
