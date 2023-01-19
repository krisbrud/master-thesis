# %%
import torch

batch_size = 10
deter_size = 15
stoch_size = 5
states = [torch.ones((batch_size, deter_size)), torch.ones((batch_size, stoch_size))]

states

is_first = torch.zeros((batch_size,))
is_first[3] = 1
is_not_first = 1 - is_first

is_not_first

state_out = [state * is_not_first.reshape(-1, 1) for state in states]
    
state_out

# %%
