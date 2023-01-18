# from ray.rllib.algorithms.dreamer import DreamerConfig
import logging
import numpy as np
import random
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

logger = logging.getLogger(__name__)
from gym_auv import DEFAULT_CONFIG



# class AuvDreamerConfig(DreamerConfig):
#     def __init__(self):
#         super().__init__()

#         # The line below is needed to be able to pass a custom model config


class AuvDreamerConfig(AlgorithmConfig):
    """Defines a configuration class from which a Dreamer Algorithm can be built.

    Example:
        >>> from ray.rllib.algorithms.dreamer import DreamerConfig
        >>> config = DreamerConfig().training(gamma=0.9, lr=0.01)\
        ...     .resources(num_gpus=0)\
        ...     .rollouts(num_rollout_workers=4)
        >>> print(config.to_dict())
        >>> # Build a Algorithm object from the config and run 1 training iteration.
        >>> trainer = config.build(env="CartPole-v1")
        >>> trainer.train()

    Example:
        >>> from ray import tune
        >>> from ray.rllib.algorithms.dreamer import DreamerConfig
        >>> config = DreamerConfig()
        >>> # Print out some default values.
        >>> print(config.clip_param)
        >>> # Update the config object.
        >>> config.training(lr=tune.grid_search([0.001, 0.0001]), clip_param=0.2)
        >>> # Set the config object's env.
        >>> config.environment(env="CartPole-v1")
        >>> # Use to_dict() to get the old-style python config dict
        >>> # when running with tune.
        >>> tune.run(
        ...     "Dreamer",
        ...     stop={"episode_reward_mean": 200},
        ...     config=config.to_dict(),
        ... )
    """

    def __init__(self):
        """Initializes a PPOConfig instance."""
        super().__init__(algo_class="AuvDreamer")

        # fmt: off
        # __sphinx_doc_begin__
        # Dreamer specific settings:
        self.td_model_lr = 6e-4
        self.actor_lr = 8e-5
        self.critic_lr = 8e-5
        self.grad_clip = 100.0
        self.lambda_ = 0.95
        self.dreamer_train_iters = 100
        self.batch_size = 50
        self.batch_length = 50
        self.imagine_horizon = 15
        self.free_nats = 3.0
        self.kl_coeff = 1.0
        self.prefill_timesteps = 100000
        self.explore_noise = 0.3

        self.dreamer_model = {
            "custom_model": DreamerModel,
            # RSSM/PlaNET parameters
            "deter_size": 200,
            "stoch_size": 30,
            # CNN Decoder Encoder
            "depth_size": 32,
            # General Network Parameters
            "hidden_size": 400,
            # Action STD
            "action_init_std": 5.0,
            
            # gym-auv specific settings
            "dense_decoder_scale": 1,  # Fixed scale parameter of gaussian in dense decoder 
            "lidar_decoder_scale": 1,  # Same, but for lidar          

            "dense_size": DEFAULT_CONFIG.sensor.dense_observation_size,
            "lidar_shape": DEFAULT_CONFIG.sensor.lidar_shape,
            "use_lidar": DEFAULT_CONFIG.sensor.use_lidar,
            "use_occupancy": DEFAULT_CONFIG.sensor.use_occupancy_grid,
            "occupancy_grid_shape": (2, DEFAULT_CONFIG.sensor.occupancy_grid_size, DEFAULT_CONFIG.sensor.occupancy_grid_size),
            "use_continuation_prediction": True,  # Also known as discount prediction

            "use_kl_balancing": True,
            "kl_balancing_alpha": 0.8,  # Only used if use_kl_balancing is True

            "entropy_coeff": 0.01,  # Coefficient for actor entropy bonus

            "tanh_squash_rewards": True,
        }

        # Override some of AlgorithmConfig's default values with PPO-specific values.
        # .rollouts()
        self.num_workers = 0  # 0
        self.num_envs_per_worker = 1
        self.horizon = 10000
        self.batch_mode = "complete_episodes"
        self.clip_actions = False

        # .training()
        self.gamma = 0.99 # 99

        print("initialized AuvDreamerConfig")
        # breakpoint()
        # # .environment()
        # self.env_config = {
        #     # Repeats action send by policy for frame_skip times in env
        #     "frame_skip": 1,
        # }

        # __sphinx_doc_end__
        # fmt: on

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        td_model_lr: Optional[float] = None,
        actor_lr: Optional[float] = None,
        critic_lr: Optional[float] = None,
        grad_clip: Optional[float] = None,
        lambda_: Optional[float] = None,
        dreamer_train_iters: Optional[int] = None,
        batch_size: Optional[int] = None,
        batch_length: Optional[int] = None,
        imagine_horizon: Optional[int] = None,
        free_nats: Optional[float] = None,
        kl_coeff: Optional[float] = None,
        prefill_timesteps: Optional[int] = None,
        explore_noise: Optional[float] = None,
        dreamer_model: Optional[dict] = None,
        **kwargs,
    ) -> "AuvDreamerConfig":
        """

        Args:
            td_model_lr: PlaNET (transition dynamics) model learning rate.
            actor_lr: Actor model learning rate.
            critic_lr: Critic model learning rate.
            grad_clip: If specified, clip the global norm of gradients by this amount.
            lambda_: The GAE (lambda) parameter.
            dreamer_train_iters: Training iterations per data collection from real env.
            batch_size: Number of episodes to sample for loss calculation.
            batch_length: Length of each episode to sample for loss calculation.
            imagine_horizon: Imagination horizon for training Actor and Critic.
            free_nats: Free nats.
            kl_coeff: KL coefficient for the model Loss.
            prefill_timesteps: Prefill timesteps.
            explore_noise: Exploration Gaussian noise.
            dreamer_model: Custom model config.

        Returns:

        """

        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if td_model_lr is not None:
            self.td_model_lr = td_model_lr
        if actor_lr is not None:
            self.actor_lr = actor_lr
        if critic_lr is not None:
            self.critic_lr = critic_lr
        if grad_clip is not None:
            self.grad_clip = grad_clip
        if lambda_ is not None:
            self.lambda_ = lambda_
        if dreamer_train_iters is not None:
            self.dreamer_train_iters = dreamer_train_iters
        if batch_size is not None:
            self.batch_size = batch_size
        if batch_length is not None:
            self.batch_length = batch_length
        if imagine_horizon is not None:
            self.imagine_horizon = imagine_horizon
        if free_nats is not None:
            self.free_nats = free_nats
        if kl_coeff is not None:
            self.kl_coeff = kl_coeff
        if prefill_timesteps is not None:
            self.prefill_timesteps = prefill_timesteps
        if explore_noise is not None:
            self.explore_noise = explore_noise
        if dreamer_model is not None:
            self.dreamer_model = dreamer_model

        return self
