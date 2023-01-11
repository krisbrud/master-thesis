# type: ignore
from typing import (
    List,
    Tuple,
    Union,
)

import logging
import ray
import numpy as np
from typing import Dict, Optional


from ray.rllib.algorithms.dreamer.utils import FreezeParameters, batchify_states
from ray.rllib.evaluation.episode import Episode
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import apply_grad_clipping
from ray.rllib.utils.typing import AgentID, TensorType
from ray.rllib.utils.annotations import override
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.models.modelv2 import restore_original_dimensions

import ray.rllib.algorithms.dreamer

import gym.spaces

from models.auv_dreamer_model import AuvDreamerModel
from models.auv_dreamer_config import AuvDreamerConfig

torch, nn = try_import_torch()
if torch:
    from torch import distributions as td
# import torch
# from torch import nn
# from torch import distributions as td

logger = logging.getLogger(__name__)


class AuvDreamerTorchPolicy(TorchPolicyV2):
    def __init__(self, observation_space, action_space, config):

        config = dict(AuvDreamerConfig().to_dict(), **config)

        TorchPolicyV2.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        # TODO: Don't require users to call this manually.
        self._initialize_loss_from_dummy_batch()
        print("initialized auv dreamer torch policy")
    

    # def _get_discount_targets(self, dones) -> torch.TensorType:
    #     """Calculates the target discount rates """

    def _get_discount_targets(self, dones: torch.TensorType, discount_rate: float):
        """Converts a tensor of boolean `done` values elementwise to discount_rate if False or 0 if True"""
        return (~dones).int() * discount_rate

    def _model_loss(self, model: ModelV2, train_batch: SampleBatch) -> Tuple[Dict[str, TensorType], TensorType]:
        obs = restore_original_dimensions(
            train_batch["obs"], self.observation_space.original_space, "torch"
        )
        # obs = train_batch["obs"]
        # PlaNET Model Loss
        latent = self.model.encoder(obs)
        post, prior = self.model.dynamics.observe(latent, train_batch["actions"])
        features = self.model.dynamics.get_feature(post)
        image_pred = self.model.decoder(features)
        reward_pred = self.model.reward(features)
        discount_pred = self.model.discount(features)
        image_loss = -torch.mean(image_pred.log_prob(train_batch["obs"].unsqueeze(1)))
        reward_loss = -torch.mean(reward_pred.log_prob(train_batch["rewards"]))
        not_dones = 1.0 - train_batch["dones"].float()

        # breakpoint()

        discount_loss = -torch.mean(discount_pred.log_prob(not_dones))
        
        # discount_target = self._get_discount_targets(dones=train_batch["dones"], discount_rate=self.config["gamma"])   # TODO

        # discount_loss = -torch.mean(discount_pred.log_prob(TODO))
        prior_dist = self.model.dynamics.get_dist(prior[0], prior[1])
        post_dist = self.model.dynamics.get_dist(post[0], post[1])

        if self.config["dreamer_model"]["use_kl_balancing"]:
            prior_dist_no_grad = self.model.dynamics.get_dist(prior[0].detach(), prior[1].detach())
            post_dist_no_grad = self.model.dynamics.get_dist(post[0].detach(), post[1].detach())

            div_no_prior_grad = torch.mean(
                torch.distributions.kl_divergence(post_dist_no_grad, prior_dist_no_grad).sum(dim=2)
            )
            div_no_post_grad = torch.mean(
                torch.distributions.kl_divergence(post_dist, prior_dist_no_grad).sum(dim=2)
            )
            alpha = self.config["dreamer_model"]["kl_balancing_alpha"]
            div = alpha * div_no_post_grad + (1 - alpha) * div_no_prior_grad
        else:
            div = torch.mean(
                torch.distributions.kl_divergence(post_dist, prior_dist).sum(dim=2)
            )
        
        div = torch.clamp(div, min=(self.config["free_nats"]))
        model_loss = self.config["kl_coeff"] * div + reward_loss + image_loss + discount_loss

        prior_ent = torch.mean(prior_dist.entropy())
        post_ent = torch.mean(post_dist.entropy())

        model_return_dict = {
            "model_loss": model_loss,
            "image_loss": image_loss,
            "reward_loss": reward_loss,
            "prior_ent": prior_ent,
            "post_ent": post_ent,
            "discount_loss": discount_loss,
            "divergence": div,
        }

        return model_return_dict, prior, post


    @override(TorchPolicyV2)
    def loss(
        self, model: ModelV2, dist_class: ActionDistribution, train_batch: SampleBatch
    ) -> Union[TensorType, List[TensorType]]:
        log_gif = False
        if "log_gif" in train_batch:
            log_gif = True

        # breakpoint()

        assert isinstance(model, AuvDreamerModel)

        # This is the computation graph for workers (inner adaptation steps)
        encoder_weights = list(self.model.encoder.parameters())
        decoder_weights = list(self.model.decoder.parameters())
        reward_weights = list(self.model.reward.parameters())
        discount_weights = list(self.model.discount.parameters())
        dynamics_weights = list(self.model.dynamics.parameters())
        critic_weights = list(self.model.value.parameters())
        model_weights = list(
            encoder_weights + decoder_weights + reward_weights + dynamics_weights + discount_weights
        )
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # if isinstance(self.observation_space.original_space, gym.spaces.Dict):
        #     # Set observation to original
        # obs = restore_original_dimensions(
        #     train_batch["obs"], self.observation_space.original_space, "torch"
        # )
        # # else:
        # #     obs = train_batch["obs"]

        ## Model Loss
        # latent = self.model.encoder(obs)
        # post, prior = self.model.dynamics.observe(latent, train_batch["actions"])
        # features = self.model.dynamics.get_feature(post)
        # image_pred = self.model.decoder(features)
        # reward_pred = self.model.reward(features)
        # # discount_pred = self.model.discount(features)
        # image_loss = -torch.mean(image_pred.log_prob(train_batch["obs"].unsqueeze(1)))
        # reward_loss = -torch.mean(reward_pred.log_prob(train_batch["rewards"]))
        
        # breakpoint()
        # # discount_target = self._get_discount_targets(dones=train_batch["dones"], discount_rate=self.config["gamma"])   # TODO

        # # discount_loss = -torch.mean(discount_pred.log_prob(TODO))
        # prior_dist = self.model.dynamics.get_dist(prior[0], prior[1])
        # post_dist = self.model.dynamics.get_dist(post[0], post[1])
        # div = torch.mean(
        #     torch.distributions.kl_divergence(post_dist, prior_dist).sum(dim=2)
        # )
        # div = torch.clamp(div, min=(self.config["free_nats"]))
        # model_loss = self.config["kl_coeff"] * div + reward_loss + image_loss
        model_return_dict, prior, post = self._model_loss(model, train_batch)
        prior_dist = self.model.dynamics.get_dist(prior[0], prior[1])
        post_dist = self.model.dynamics.get_dist(post[0], post[1])
        #
        ## Actor Loss
        # [imagine_horizon, batch_length*batch_size, feature_size]
        with torch.no_grad():
            actor_states = [v.detach() for v in post]
        with FreezeParameters(model_weights):
            imag_feat = self.model.imagine_ahead(
                actor_states, self.config["imagine_horizon"]
            )

            # Give a bonus for higher entropy in the policy
            entropy_loss = -torch.mean(model.actor(imag_feat).base_dist.base_dist.entropy()) * self.config["dreamer_model"]["entropy_coeff"]
        with FreezeParameters(model_weights + critic_weights):
            reward = self.model.reward(imag_feat).mean
            value = self.model.value(imag_feat).mean
            # if config["use_discount_prediction"]
            # discount = self.model.discount(imag_feat).mean * self.config["gamma"]
            
            # # We predict whether this timestep is done, i.e. if the next will be 
            # # Pad discount prediction with actual values for first time step
            # first_not_done = 1.0 - train_batch[SampleBatch.DONES].reshape(1, -1).float()  # shape: (1, batch_size)
            
            # # Shift the discount rates - as they measure whether the following state
            # # will be valid, not if the current state is valid.
            # # Pad on beginning with whether the first state in the replay buffer is terminal
            # padded_discount = torch.cat((first_not_done, discount[:-1]))
            # discount_cumprod = torch.cumprod(padded_discount, dim=0)

        pcont = self.config["gamma"] * torch.ones_like(reward)

        # As in the implementation of DreamerV2, we override the predicted discount rate of the first timestep with the true
        # discount rate from the replay buffer. 
        # We do this because this value is known, while the future imagined timesteps are just predictions
        # first_is_done = 1.0 - train_batch["dones"][:1].int()
        
        # Estimate probability of continuing
        # prob_continue = padded_discount  # discount  #  pcont
        prob_continue = pcont

        # Similar to GAE-Lambda, calculate value targets
        # next_values = torch.cat([value[:-1][1:], value[-1][None]], dim=0)  # This is equivalent to value[1:]
        next_values = value[1:] 
        
        # breakpoint()
        # print(f"{reward.shape = }")
        # print(f"{prob_continue.shape = }")
        # print(f"{next_values.shape = }")

        # The inputs variable contains the rewards (except the last one) as well as the probability of continuing
        # multiplied element-wise with next_values (essentially all the values except the first one)
        inputs = reward[:-1] + prob_continue[:-1] * next_values * (1 - self.config["lambda"])

        def agg_fn(last, step_inputs, prob_continue):
            # Essentially a variant of a reducer for calculating value targets.
            # y[0] are the inputs (defined above), while y[1] is typically the probability of continuing
            return step_inputs + prob_continue * self.config["lambda"] * last

        last = value[-1]
        returns = []
        for i in reversed(range(len(inputs))):
            last = agg_fn(last, inputs[i], prob_continue[:-1][i])
            returns.append(last)

        returns = list(reversed(returns))
        returns = torch.stack(returns, dim=0)
        discount_shape = prob_continue[:1].size()
        discount = torch.cumprod(
            torch.cat([torch.ones(*discount_shape).to(device), prob_continue[:-2]], dim=0),
            dim=0,
        )
        actor_loss = -torch.mean(discount * returns) # + entropy_loss
        # print(f"{discount_cumprod.shape = }")
        # print(f"{discount_cumprod[:-1].shape = }")
        # print(f"{returns.shape = }")


        # breakpoint()

        # actor_loss = -torch.mean(discount_cumprod[:-1] * returns) + entropy_loss

        # Critic Loss
        with torch.no_grad():
            val_feat = imag_feat.detach()[:-1]
            target = returns.detach()
            # val_discount = discount.detach()
            val_discount = discount_cumprod[:-1].detach()
        val_pred = self.model.value(val_feat)

        # breakpoint()

        critic_loss = -torch.mean(val_discount * val_pred.log_prob(target))

        # Logging purposes
        prior_ent = torch.mean(prior_dist.entropy())
        post_ent = torch.mean(post_dist.entropy())
        gif = None
        # if log_gif:
        #     gif = log_summary(
        #         train_batch["obs"],
        #         train_batch["actions"],
        #         latent,
        #         image_pred,
        #         self.model,
        #     )

        return_dict = {
            # "model_loss": model_loss,
            # "reward_loss": reward_loss,
            # "image_loss": image_loss,
            # "divergence": div,
            **model_return_dict, # model_loss, image_loss, divergence etc.
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "prior_ent": prior_ent,
            "post_ent": post_ent,
        }

        for key, val in return_dict.items():
            model.tower_stats[key] = val
        
        if gif is not None:
            return_dict["log_gif"] = gif
        self.stats_dict = return_dict

        loss_dict = self.stats_dict

        # breakpoint()

        return (
            loss_dict["model_loss"],
            loss_dict["actor_loss"],
            loss_dict["critic_loss"],
        )

    @override(TorchPolicyV2)
    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[
            Dict[AgentID, Tuple["Policy", SampleBatch]]
        ] = None,
        episode: Optional["Episode"] = None,
    ) -> SampleBatch:
        """Batch format should be in the form of (o_t, a_(t-1), r_t, done_t)
        When t=0, the resetted obs is paired with action and reward of 0, as well as an initial done of false.
        """

        # breakpoint()
        # print("Done at end of trajectory before overwrite: ", sample_batch[SampleBatch.DONES][-1])
        
        # Overwrite the done at the end of the trajectory to be False if collision_or_reached_goal is False
        # This is because we only want the discount predictor to estimate these done values, not the ones from the horizon
        
        if isinstance(sample_batch["infos"][-1], dict):  # During setup, sample_batch["infos"] are all 0.0.
            try:
                if sample_batch["infos"][-1]["collision_or_reached_goal"] == False:
                    sample_batch[SampleBatch.DONES][-1] = False
            except KeyError:
                print("Warning [AuvDreamerTorchPolicy.post_process_trajectory]: Trying to access infos['collision_or_reached_goal'] but it doesn't exist.")
            except Exception as e:
                print("Error [AuvDreamerTorchPolicy.post_process_trajectory]: ", e)

        # print("Done at end of trajectory after overwrite: ", sample_batch[SampleBatch.DONES][-1]) 

        obs = sample_batch[SampleBatch.OBS]
        new_obs = sample_batch[SampleBatch.NEXT_OBS]
        action = sample_batch[SampleBatch.ACTIONS]
        reward = sample_batch[SampleBatch.REWARDS]
        eps_ids = sample_batch[SampleBatch.EPS_ID]
        dones = sample_batch[SampleBatch.DONES]
        # print("postprocessing trajectory!")

        act_shape = action.shape
        act_reset = np.array([0.0] * act_shape[-1])[None]
        rew_reset = np.array(0.0)[None]
        dones_reset = np.array(False)[None]
        obs_end = np.array(new_obs[act_shape[0] - 1])[None]

        batch_obs = np.concatenate([obs, obs_end], axis=0)
        batch_action = np.concatenate([act_reset, action], axis=0)
        batch_rew = np.concatenate([rew_reset, reward], axis=0)
        batch_eps_ids = np.concatenate([eps_ids, eps_ids[-1:]], axis=0)
        batch_dones = np.concatenate([dones_reset, dones])

        new_batch = {
            SampleBatch.OBS: batch_obs,
            SampleBatch.REWARDS: batch_rew,
            SampleBatch.ACTIONS: batch_action,
            SampleBatch.EPS_ID: batch_eps_ids,
            SampleBatch.DONES: batch_dones,
        }
        return SampleBatch(new_batch)

    def stats_fn(self, train_batch):
        return convert_to_numpy(
            {
                "model_loss": torch.mean(torch.stack(self.get_tower_stats("model_loss"))),
                "reward_loss": torch.mean(torch.stack(self.get_tower_stats("reward_loss"))),
                "image_loss": torch.mean(torch.stack(self.get_tower_stats("image_loss"))),
                "discount_loss": torch.mean(torch.stack(self.get_tower_stats("discount_loss"))),
                "divergence": torch.mean(torch.stack(self.get_tower_stats("divergence"))),
                "actor_loss": torch.mean(torch.stack(self.get_tower_stats("actor_loss"))),
                "critic_loss": torch.mean(torch.stack(self.get_tower_stats("critic_loss"))),
                "prior_ent": torch.mean(torch.stack(self.get_tower_stats("prior_ent"))),
                "post_ent": torch.mean(torch.stack(self.get_tower_stats("post_ent"))),
            }
        )


    @override(TorchPolicyV2)
    def optimizer(self):
        model = self.model
        encoder_weights = list(model.encoder.parameters())
        decoder_weights = list(model.decoder.parameters())
        reward_weights = list(model.reward.parameters())
        dynamics_weights = list(model.dynamics.parameters())
        actor_weights = list(model.actor.parameters())
        critic_weights = list(model.value.parameters())
        model_opt = torch.optim.Adam(
            encoder_weights + decoder_weights + reward_weights + dynamics_weights,
            lr=self.config["td_model_lr"],
        )
        actor_opt = torch.optim.Adam(actor_weights, lr=self.config["actor_lr"])
        critic_opt = torch.optim.Adam(critic_weights, lr=self.config["critic_lr"])

        return (model_opt, actor_opt, critic_opt)

    def action_sampler_fn(policy, model, obs_batch, state_batches, explore, timestep):
        """Action sampler function has two phases. During the prefill phase,
        actions are sampled uniformly [-1, 1]. During training phase, actions
        are evaluated through DreamerPolicy and an additive gaussian is added
        to incentivize exploration.
        """
        obs = obs_batch["obs"]

        if isinstance(obs, dict):
            bsize = 1
            logger.warn(
                "Assuming batch size 1 since observation is dictionary in optimizer!"
            )
        else:
            bsize = obs.shape[0]

        # Custom Exploration
        if timestep <= policy.config["prefill_timesteps"]:
            logp = None
            # Random action in space [-1.0, 1.0]
            eps = torch.rand(1, model.action_space.shape[0], device=obs.device)
            action = 2.0 * eps - 1.0
            state_batches = model.get_initial_state()
            # batchify the intial states to match the batch size of the obs tensor
            state_batches = batchify_states(state_batches, bsize, device=obs.device)
        else:
            # Weird RLlib Handling, this happens when env rests
            if len(state_batches) == 0:

                state_batches = AuvDreamerTorchPolicy._manually_reset_state_batches(model, obs, bsize)
            elif len(state_batches[0].size()) == 3:
                state_batches = AuvDreamerTorchPolicy._manually_reset_state_batches(model, obs, bsize)
                # # Very hacky, but works on all envs
                # state_batches = model.get_initial_state().to(device=obs.device)
                # # batchify the intial states to match the batch size of the obs tensor
                # state_batches = batchify_states(state_batches, bsize, device=obs.device)
            action, logp, state_batches = model.policy(obs, state_batches, explore)
            action = td.Normal(action, policy.config["explore_noise"]).sample()
            action = torch.clamp(action, min=-1.0, max=1.0)

        # policy.global_timestep += policy.config["action_repeat"]

        return action, logp, state_batches

    def _manually_reset_state_batches(model, obs, bsize):
        # For reconciling when state batches are not in the expected format

        # Very hacky, but works on all envs
        state_batches_list = model.get_initial_state()
        state_batches = [x.to(device=obs.device) for x in state_batches_list]

        # batchify the intial states to match the batch size of the obs tensor
        state_batches = batchify_states(state_batches, bsize, device=obs.device)
        return state_batches 

    def make_model(self):

        model = ModelCatalog.get_model_v2(
            self.observation_space,
            self.action_space,
            1,
            self.config["dreamer_model"],
            name="DreamerModel",
            framework="torch",
        )

        self.model_variables = model.variables()

        return model

    def extra_grad_process(
        self, optimizer: "torch.optim.Optimizer", loss: TensorType
    ) -> Dict[str, TensorType]:
        return apply_grad_clipping(self, optimizer, loss)


# Creates gif
def log_summary(obs, action, embed, image_pred, model):
    truth = obs[:6] + 0.5
    recon = image_pred.mean[:6]
    init, _ = model.dynamics.observe(embed[:6, :5], action[:6, :5])
    init = [itm[:, -1] for itm in init]
    prior = model.dynamics.imagine(action[:6, 5:], init)
    openl = model.decoder(model.dynamics.get_feature(prior)).mean

    mod = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (mod - truth + 1.0) / 2.0
    return torch.cat([truth, mod, error], 3)
