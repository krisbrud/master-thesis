# %% 
import math 

def output_size(input_size, kernel_size, stride):
    return math.floor((input_size - kernel_size) / stride) + 1

o = 64 # 256
k = 4
s = 2
for _ in range(4):
    o = output_size(o, k, s)
    print(o)


# %%
import torch
from torch import nn
# %%
transpose_1d_net = nn.Sequential(
    nn.ConvTranspose1d(1024, 1, kernel_size=5, stride=2),
    nn.ConvTranspose1d(1, 1, kernel_size=5, stride=2),
    nn.ConvTranspose1d(1, 1, kernel_size=5, stride=2),
    nn.ConvTranspose1d(1, 1, kernel_size=5, stride=2),
    nn.ConvTranspose1d(1, 1, kernel_size=6, stride=2),
    nn.ConvTranspose1d(1, 1, kernel_size=6, stride=2),
)

x = torch.zeros((1024, 1))
out = transpose_1d_net(x)
print(out.shape)

# %%
depth = 32
layers = [
    nn.Conv2d(3, depth, 4, stride=2),
    nn.Conv2d(depth, 2 * depth, 4, stride=2),
    nn.Conv2d(2 * depth, 4 * depth, 4, stride=2),
    nn.Conv2d(4 * depth, 8 * depth, 4, stride=2),
]
convnet = nn.Sequential(*layers)
x = torch.zeros((7, 3, 64, 64))
convout = convnet(x)
# %%
