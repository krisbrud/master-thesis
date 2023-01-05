import pytest
import torch
from ray.rllib.algorithms.dreamer.dreamer_model import RSSM

action_size = 2
embed_size = 1024
stoch_size = 30
deter_size = 200
hidden_size = 200

@pytest.fixture
def rssm() -> RSSM:
    rssm = RSSM(action_size, embed_size, stoch_size, deter_size, hidden_size)
    return rssm

def test_rssm_device(rssm):
    assert str(rssm.device) == "cuda"


def test_batched_observe(rssm: RSSM):
    batch_size = 7
    embed_size = 1024
    embedding = torch.zeros((batch_size, embed_size)).to("cuda")
    action = torch.zeros((batch_size, action_size)).to("cuda")
    rssm.cuda()
    post, prior = rssm.observe(embedding, action)
    assert post[0].shape == (batch_size, 1, stoch_size)

if __name__ == "__main__":
    rssm = RSSM(action_size, embed_size, stoch_size, deter_size, hidden_size)
    test_batched_observe(rssm)

