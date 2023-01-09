import pytest
import numpy as np

from models.episode_replay_buffer import EpisodeSequenceBuffer
from ray.rllib.policy.sample_batch import SampleBatch

@pytest.fixture
def episode_sequence_buffer() -> EpisodeSequenceBuffer:
    return EpisodeSequenceBuffer()

def test_add_and_retrieve(episode_sequence_buffer: EpisodeSequenceBuffer):
    dones = [False] * 999
    dones.append(True)
    sample_batch = SampleBatch({
        SampleBatch.OBS: np.zeros((1000, 3)),
        SampleBatch.ACTIONS: np.zeros((1000, 2)),
        SampleBatch.EPS_ID: np.zeros((1000,), dtype=np.int_),
        SampleBatch.DONES: np.array(dones)
    })
    episode_sequence_buffer.add(sample_batch)
    
    batch = episode_sequence_buffer.sample(50)

def test_add_and_retrieve_no_done_at_end(episode_sequence_buffer: EpisodeSequenceBuffer):
    dones = [False] * 1000
    sample_batch = SampleBatch({
        SampleBatch.OBS: np.zeros((1000, 3)),
        SampleBatch.ACTIONS: np.zeros((1000, 2)),
        SampleBatch.EPS_ID: np.zeros((1000,), dtype=np.int_),
        SampleBatch.DONES: np.array(dones)
    })
    episode_sequence_buffer.add(sample_batch)
    
    batch = episode_sequence_buffer.sample(50)


if __name__ == "__main__":
    test_add_and_retrieve(EpisodeSequenceBuffer())
    test_add_and_retrieve_no_done_at_end(EpisodeSequenceBuffer())