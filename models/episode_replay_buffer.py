import random

import numpy as np

from ray.rllib.utils.replay_buffers import ReplayBuffer, StorageUnit
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.policy.sample_batch import concat_samples, SampleBatch

class EpisodeSequenceBuffer(ReplayBuffer):
    def __init__(self, capacity: int = 1000, replay_sequence_length: int = 50):
        """Stores episodes and samples sequences of size `replay_sequence_length`.
        Args:
            capacity: Maximum number of episodes this buffer can store
            replay_sequence_length: Episode chunking length in sample()
        """
        super().__init__(capacity=capacity, storage_unit=StorageUnit.EPISODES)
        self.replay_sequence_length = replay_sequence_length

    def add(self, batch: SampleBatchType, **kwargs):
        """Adds a batch of episodes to the buffer
        Args:
            batch: Batch of episodes to be added
        """
        # We overwrite this method to allow adding methods where "done = False" at the end.
        # This is because we only want done to be True in the case of solving the environment or failing by collision 
        # (not due to time out etc)

        if not batch.count > 0:
            return

        assert isinstance(batch, SampleBatch)

        for eps in batch.split_by_episode():
            # Only add full episodes to the buffer
            # Check only if info is available
            self._add_single_batch(eps, **kwargs)
            

    def sample(self, num_items: int):
        """Samples [batch_size, length] from the list of episodes
        Args:
            num_items: batch_size to be sampled
        """
        episodes_buffer = []
        while len(episodes_buffer) < num_items:
            episode = super().sample(1)
            if episode.count < self.replay_sequence_length:
                continue
            available = episode.count - self.replay_sequence_length

            # has_done_at_end = episode["dones"][-1]
            # should_sample_last_sequence = random.random() < self.replay_sequence_length / max(available, 1)
            # if has_done_at_end and should_sample_last_sequence:
            #     # Sample the last sequence of the episode with probability self.replay_sequence_length / available
            #     # to avoid sampling the "dones" too little
            #     episodes_buffer.append(
            #         episode[available:]  # Last replay_sequence_length items
            #     )
            # else:
            index = int(random.randint(0, available))
            
            episode_segment = episode[index : index + self.replay_sequence_length]
            is_firsts = np.zeros((self.replay_sequence_length,))
            is_firsts[0] = 1
            episode_segment["is_firsts"] = is_firsts

            episodes_buffer.append(episode_segment)

        return concat_samples(episodes_buffer)
