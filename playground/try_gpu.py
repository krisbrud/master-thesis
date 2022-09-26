import torch

x = torch.Tensor(list(range(10)))
x.to("cuda")
