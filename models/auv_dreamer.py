from typing import Callable, Union
from ray.rllib.algorithms.dreamer import Dreamer, DreamerConfig
import gym

from models.auv_dreamer_model import AuvDreamerModel
from models.auv_dreamer_torch_policy import AuvDreamerTorchPolicy
from models.auv_dreamer_config import AuvDreamerConfig


import logging
import numpy as np
import gym_auv

# import random
from typing import Optional

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dreamer.dreamer_torch_policy import DreamerTorchPolicy
from ray.rllib.execution.common import STEPS_SAMPLED_COUNTER, _get_shared_metrics
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, concat_samples
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.algorithms.dreamer.dreamer_model import DreamerModel
from ray.rllib.execution.rollout_ops import (
    ParallelRollouts,
    synchronous_parallel_sample,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import Deprecated
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO
from ray.rllib.utils.typing import (
    PartialAlgorithmConfigDict,
    SampleBatchType,
    AlgorithmConfigDict,
    ResultDict,
)

from ray.rllib.algorithms.dreamer.dreamer import (
    EpisodicBuffer,
    total_sampled_timesteps,
)

from pipeline.config import get_auv_dreamer_config_dict


logger = logging.getLogger(__name__)


class AuvDreamer(Algorithm):
    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return AuvDreamerConfig().to_dict()

    @override(Algorithm)
    def validate_config(self, config: AlgorithmConfigDict) -> None:
        # Call super's validation method.
        super().validate_config(config)

        config["action_repeat"] = config["env_config"]["frame_skip"]
        if config["num_gpus"] > 1:
            raise ValueError("`num_gpus` > 1 not yet supported for Dreamer!")
        if config["framework"] != "torch":
            raise ValueError("Dreamer not supported in Tensorflow yet!")
        if config["batch_mode"] != "complete_episodes":
            raise ValueError("truncate_episodes not supported")
        if config["num_workers"] != 0:
            raise ValueError("Distributed Dreamer not supported yet!")
        if config["clip_actions"]:
            raise ValueError("Clipping is done inherently via policy tanh!")
        if config["dreamer_train_iters"] <= 0:
            raise ValueError(
                "`dreamer_train_iters` must be a positive integer. "
                f"Received {config['dreamer_train_iters']} instead."
            )
        if config["action_repeat"] > 1:
            config["horizon"] = config["horizon"] / config["action_repeat"]

    @override(Algorithm)
    def get_default_policy_class(self, config: AlgorithmConfigDict):
        return AuvDreamerTorchPolicy

    @override(Algorithm)
    def setup(self, config: PartialAlgorithmConfigDict):
        super().setup(config)
        # `training_iteration` implementation: Setup buffer in `setup`, not
        # in `execution_plan` (deprecated).

        if self.config["_disable_execution_plan_api"] is True:
            self.local_replay_buffer = EpisodicBuffer(length=config["batch_length"])

            # Prefill episode buffer with initial exploration (uniform sampling)
            while (
                total_sampled_timesteps(self.workers.local_worker())
                < self.config["prefill_timesteps"]
            ):
                samples = self.workers.local_worker().sample()
                self.local_replay_buffer.add(samples)
        else:
            raise ValueError(
                "`_disable_execution_api` is set to False, which is not supported"
            )

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        local_worker = self.workers.local_worker()

        # Number of sub-iterations for Dreamer
        dreamer_train_iters = self.config["dreamer_train_iters"]
        batch_size = self.config["batch_size"]

        # Collect SampleBatches from rollout workers.
        # new_sample_batches = synchronous_parallel_sample(worker_set=self.workers)
        batch = synchronous_parallel_sample(worker_set=self.workers)
        # for batch in new_sample_batches:
        self._counters[NUM_AGENT_STEPS_SAMPLED] += batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += batch.env_steps()
        self.local_replay_buffer.add(batch)
        # self.local_replay_buffer.add(new_sample_batches)

        fetches = {}

        # Dreamer training loop.
        # Run multiple sub-iterations for each training iteration.
        print("Starting training iteration!")
        for n in range(dreamer_train_iters):
            # print(f"sub-iteration={n}/{dreamer_train_iters}")
            batch = self.local_replay_buffer.sample(batch_size)
            fetches = local_worker.learn_on_batch(batch)
        print("Training iteration done!")

        if fetches:
            # Custom logging.
            policy_fetches = fetches[DEFAULT_POLICY_ID]["learner_stats"]
            if "log_gif" in policy_fetches:
                gif = policy_fetches["log_gif"]
                policy_fetches["log_gif"] = self._postprocess_gif(gif)

        return fetches


def _get_auv_dreamer_model_options() -> dict:
    model_config = {
        "custom_model": AuvDreamerModel,
        # RSSM/PlaNET parameters
        "deter_size": 32,  # 200,
        "stoch_size": 3,  # 30,
        # CNN Decoder Encoder
        "depth_size": 32,
        # General Network Parameters
        "hidden_size": 32,  # 400,
        # Action STD
        "action_init_std": 5.0,
    }
    return model_config


def auv_dreamer_factory(
    env_name: str, env_config: Union[dict, None] = None, num_envs_per_worker=None
) -> AuvDreamer:
    """Instantiates the Dreamer algorithm with a default
    model suitable for the gym-auv environment"""
    dreamer_config = get_auv_dreamer_config_dict(
        env_name=env_name, env_config=env_config
    )

    # Instantiate the algorithm
    # dreamer = Dreamer(config=dreamer_config)
    dreamer = AuvDreamer(config=dreamer_config)
    return dreamer


# if __name__ == "__main__":
#     # TODO: Register the env
#     auv_dreamer_factory("TestScenario1-v0")
