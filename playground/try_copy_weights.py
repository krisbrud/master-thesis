# %%

import torch
import torch.nn as nn
import copy

foo = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 10))

bar = copy.deepcopy(foo)
# %%
