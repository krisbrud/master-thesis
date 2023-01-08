import random

from ray.rllib.utils.replay_buffers import ReplayBuffer, StorageUnit
from ray.rllib.policy.sample_batch import concat_samples

class EpisodeSequenceBuffer(ReplayBuffer):
    def __init__(self, capacity: int = 1000, replay_sequence_length: int = 50):
        """Stores episodes and samples sequences of size `replay_sequence_length`.
        Args:
            capacity: Maximum number of episodes this buffer can store
            replay_sequence_length: Episode chunking length in sample()
        """
        super().__init__(capacity=capacity, storage_unit=StorageUnit.EPISODES)
        self.replay_sequence_length = replay_sequence_length

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

            has_done_at_end = episode["dones"][-1]
            should_sample_last_sequence = random.random() < self.replay_sequence_length / available
            if has_done_at_end and should_sample_last_sequence:
                # Sample the last sequence of the episode with probability self.replay_sequence_length / available
                # to avoid sampling the "dones" too little
                episodes_buffer.append(
                    episode[-self.replay_sequence_length:]  # Last replay_sequence_length items
                )
            else:
                index = int(random.randint(0, available))
                episodes_buffer.append(episode[index : index + self.replay_sequence_length])

        return concat_samples(episodes_buffer)
