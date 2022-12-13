# %% 
import numpy as np

from models.episode_replay_buffer import EpisodeSequenceBuffer
from ray.rllib.policy.sample_batch import SampleBatch

episode_buffer = EpisodeSequenceBuffer()
dones = [False] * 99
dones.append(True)
dones = dones * 2

mock_batch = SampleBatch({
    "obs": list(range(200)),
    "actions": list(reversed(list(range(200)))),
    "dones": dones,
    "eps_ids": ([0] * 100) + ([1] * 100),
})


episode_buffer.add(mock_batch)
print(episode_buffer)

# %%

## Check to what extent the dones (which there are two of) are sampled
# These will usually only be available by picking one index each, and are much rarer to sample than
# most other samples in the buffer.

n_dones = 0

for i in range(100):
    batch = episode_buffer.sample(1)

    if any(batch["dones"]):
        n_dones += 1

print(n_dones)