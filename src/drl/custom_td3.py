"""Custom TD3 policy with LayerNorm on critic networks (DAWN paper).

LayerNorm after each hidden layer stabilizes Q-value scale during training,
preventing the critic destabilization observed in C006 Phase 2 (actor loss
tripled 262→730 as Q-values diverged).

Architecture change is critic-only — actor MLP is unchanged.

Reference:
    Lyle et al., "Understanding Plasticity in Neural Networks" (ICML 2023)
    — LayerNorm prevents loss of plasticity in deep RL critics.

SB3 version pin: This module is tightly coupled to SB3 2.3.2's
ContinuousCritic internals. The version assertion will catch upgrades
that might break the constructor bypass.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import numpy as np
import stable_baselines3 as sb3
import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.policies import BaseModel, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.td3 import TD3
from stable_baselines3.td3.policies import TD3Policy

assert sb3.__version__ == "2.3.2", (
    f"custom_td3.py requires SB3 2.3.2, got {sb3.__version__}. "
    f"ContinuousCritic internals may have changed — review before updating."
)


def create_mlp_layernorm(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
    with_bias: bool = True,
) -> List[nn.Module]:
    """Build an MLP with LayerNorm after each hidden Linear layer.

    Same signature as ``stable_baselines3.common.torch_layers.create_mlp``
    but inserts ``nn.LayerNorm(hidden_dim)`` between each Linear and its
    activation. The output layer gets NO LayerNorm (only hidden layers).

    Pattern per hidden layer: Linear → LayerNorm → activation_fn
    Output layer: Linear [→ Tanh if squash_output]
    """
    modules: List[nn.Module] = []

    if len(net_arch) > 0:
        modules.append(nn.Linear(input_dim, net_arch[0], bias=with_bias))
        modules.append(nn.LayerNorm(net_arch[0]))
        modules.append(activation_fn())

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1], bias=with_bias))
        modules.append(nn.LayerNorm(net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim, bias=with_bias))
    if squash_output:
        modules.append(nn.Tanh())

    return modules


class LayerNormContinuousCritic(ContinuousCritic):
    """ContinuousCritic with LayerNorm after each hidden layer.

    Bypasses ``ContinuousCritic.__init__()`` (which calls ``create_mlp``
    without LayerNorm) and replicates the Q-network construction loop
    with ``create_mlp_layernorm`` instead.

    ``forward()`` and ``q1_forward()`` are inherited unchanged — they
    iterate ``self.q_networks`` which we populate identically.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        # Skip ContinuousCritic.__init__ — call grandparent BaseModel directly.
        # BaseModel.__init__ sets observation_space, action_space,
        # features_extractor, normalize_images.
        BaseModel.__init__(
            self,
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks: List[nn.Module] = []
        for idx in range(n_critics):
            q_net_list = create_mlp_layernorm(
                features_dim + action_dim, 1, net_arch, activation_fn,
            )
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class LayerNormTD3Policy(TD3Policy):
    """TD3Policy that uses LayerNormContinuousCritic for the critic.

    Actor creation is unchanged — no LayerNorm on actor (matches plan:
    critic-only stabilization).
    """

    def make_critic(
        self, features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> LayerNormContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor,
        )
        critic = LayerNormContinuousCritic(**critic_kwargs).to(self.device)
        print(
            "[CRITIC_LN] Constructed LayerNormContinuousCritic via "
            "LayerNormTD3Policy.make_critic()",
            flush=True,
        )
        return critic

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data["critic_layernorm"] = True
        return data


class CustomClippedTD3(TD3):
    """TD3 with critic gradient clipping.

    SB3 2.3.2's TD3.train() has NO gradient clipping on either optimizer.
    This subclass overrides train() to add clip_grad_norm_ on the critic
    before optimizer.step(). Actor is NOT clipped.

    Composes with LayerNormTD3Policy (Policy vs Algorithm subclass).

    Phase 5.1: Prevents critic loss divergence observed in C006 Phase 5
    (critic_loss=107.77, Q1_mean -0.44→-94.4).
    """

    def __init__(self, *args, max_grad_norm_critic: float = 0.5,
                 caps_lambda_t: float = 0.0,
                 caps_actor_lambda: float = 0.0, **kwargs):
        self.caps_lambda_t = caps_lambda_t
        self.caps_actor_lambda = caps_actor_lambda
        self.max_grad_norm_critic = max_grad_norm_critic
        self._last_action_stats = None
        import stable_baselines3 as _sb3
        assert _sb3.__version__ == "2.3.2", (
            f"_sample_action override requires SB3 2.3.2; "
            f"got {_sb3.__version__}. Review override before use."
        )
        super().__init__(*args, **kwargs)

    # MIRRORED FROM SB3 2.3.2:
    # stable_baselines3/common/off_policy_algorithm.py lines 360-407
    # If SB3 version changes, re-verify contract.
    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        ):
            unscaled_action = np.array(
                [self.action_space.sample() for _ in range(n_envs)]
            )
        else:
            assert self._last_obs is not None
            unscaled_action, _ = self.predict(
                self._last_obs, deterministic=False)

        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            if action_noise is not None:
                pre_clip = scaled_action + action_noise()
            else:
                pre_clip = scaled_action

            abs_pre = np.abs(pre_clip)
            self._last_action_stats = {
                "raw_action_max_per_env": np.max(abs_pre, axis=-1).astype(
                    float),
                "boundary_hit_per_env": (abs_pre >= 1.0 - 1e-6).any(
                    axis=-1),
                "overflow_hit_per_env": (abs_pre > 1.0 + 1e-6).any(
                    axis=-1),
            }

            scaled_action = np.clip(pre_clip, -1.0, 1.0)
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            buffer_action = unscaled_action
            action = buffer_action
            self._last_action_stats = None

        return action, buffer_action

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Verbatim SB3 2.3.2 TD3.train() with critic gradient clipping.

        Single insertion: th.nn.utils.clip_grad_norm_ before critic.optimizer.step().
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses, caps_losses = [], [], []
        critic_grad_norms = []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            # Phase 7A: Huber loss (smooth_l1 with beta=1.0) caps gradient at 1.0 for
            # large TD errors, preventing single-batch destabilization from extreme
            # transitions that caused Phase 7 critic oscillation (loss cycling 45-120).
            critic_loss = sum(F.smooth_l1_loss(current_q, target_q_values, beta=1.0) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            # Capture pre-clip gradient norm for monitoring
            _pre_clip_norm = th.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm_critic)
            critic_grad_norms.append(_pre_clip_norm.item())
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                actions_pi = self.actor(replay_data.observations)
                actor_loss = -self.critic.q1_forward(replay_data.observations, actions_pi).mean()

                # P1-B: CAPS temporal smoothness (Mysore et al., ICRA 2021)
                _effective_caps = self.caps_lambda_t + self.caps_actor_lambda
                if _effective_caps > 0.0:
                    next_actions_pi = self.actor(replay_data.next_observations)
                    caps_loss = _effective_caps * th.mean(
                        (actions_pi - next_actions_pi) ** 2)
                    actor_loss = actor_loss + caps_loss
                    caps_losses.append(caps_loss.item())

                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(critic_grad_norms) > 0:
            self._critic_grad_norm_mean = float(np.mean(critic_grad_norms))
            self._critic_grad_norm_max = float(np.max(critic_grad_norms))
            self.logger.record("train/critic_grad_norm_mean", self._critic_grad_norm_mean)
            self.logger.record("train/critic_grad_norm_max", self._critic_grad_norm_max)
        if len(caps_losses) > 0:
            self.logger.record("train/caps_loss", np.mean(caps_losses))
