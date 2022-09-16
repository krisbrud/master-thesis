import torch
from torch import nn

batch_size = 12
size_a = 5
size_b = 7

a = torch.rand((batch_size, size_a))
print(f"{a.shape = }")

b = torch.rand((batch_size, size_b))
print(f"{b.shape = }")

c = torch.cat((a, b), dim=1)
print(f"{c.shape = }")
