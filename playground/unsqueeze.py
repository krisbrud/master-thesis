# %%
# Testing the unsqueeze functionality, to be used in AuvDreamerPolicy
import torch

x = torch.rand((13, 50))
foo = x.unsqueeze(1)
print(foo.shape)
# %%
