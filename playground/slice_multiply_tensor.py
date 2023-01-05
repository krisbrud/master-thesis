# Try to find out why bug where weird bug happens
# when running code but not in debugger
import torch

foo = torch.ones((15, 33))
bar = torch.ones((14, 33))

loss = -torch.mean(foo[:-1] * bar)