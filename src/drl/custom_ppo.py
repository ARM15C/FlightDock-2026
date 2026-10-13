"""SeparateClipPPO — PPO with independent actor/critic gradient clipping.

C005b Commit 3: Fixes the gradient monopoly discovered in C005a, where the
value function's ~5000 loss dominated the combined gradient vector and the
single clip_grad_norm_ attenuated policy gradients to ~0.003.

This subclass overrides PPO.train() to clip actor and critic gradients
independently, each to max_grad_norm. The policy gradient now gets its
full clip budget regardless of value gradient magnitude.

Maintenance note: This copies PPO.train() verbatim from SB3 2.3.2 with
only the gradient clipping block replaced. Pin SB3 version in requirements
to prevent silent breakage.
"""

from __future__ import annotations

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

import stable_baselines3 as _sb3
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance

_EXPECTED_SB3_VERSION = "2.3.2"
assert _sb3.__version__ == _EXPECTED_SB3_VERSION, (
    f"SeparateClipPPO.train() was copied from SB3 {_EXPECTED_SB3_VERSION}. "
    f"Installed: {_sb3.__version__}. Re-copy PPO.train() before using."
)


class SeparateClipPPO(PPO):
    """PPO with separate actor/critic gradient clipping.

    Identical to SB3 PPO except:
    1. Gradient clipping is applied independently to actor and critic params
    2. Per-group gradient norms are logged as train/actor_grad_norm and
       train/critic_grad_norm
    3. Last grad norms stored as instance attributes for callback access
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Instance attributes for callback access (set after first train())
        self.last_actor_grad_norm: float = float("nan")
        self.last_critic_grad_norm: float = float("nan")

    def train(self) -> None:
        """Update policy using the currently gathered rollout buffer.

        Copied from SB3 2.3.2 PPO.train() (ppo.py:184-304) with gradient
        clipping on line 281 replaced by separate actor/critic clipping.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} "
                            f"due to reaching max kl: "
                            f"{approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()

                # ---- C005b: Separate actor/critic gradient clipping ----
                # Original SB3 line 281:
                #   th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # This clips ALL params together, letting value gradient (~5000 loss)
                # consume the entire clip budget. With separate clipping, each group
                # gets its own max_grad_norm budget.
                actor_params = (
                    list(self.policy.mlp_extractor.policy_net.parameters())
                    + list(self.policy.action_net.parameters())
                    + [self.policy.log_std]
                )
                critic_params = (
                    list(self.policy.mlp_extractor.value_net.parameters())
                    + list(self.policy.value_net.parameters())
                )
                # NOTE: clip_grad_norm_() returns the PRE-CLIP total norm (PyTorch convention).
                # The actual gradient IS clipped to max_grad_norm for the optimizer step.
                # Logged values show pre-clip magnitude, NOT the applied gradient magnitude.
                actor_grad_norm = th.nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
                critic_grad_norm = th.nn.utils.clip_grad_norm_(critic_params, self.max_grad_norm)

                # Store for callback access
                self.last_actor_grad_norm = actor_grad_norm.item()
                self.last_critic_grad_norm = critic_grad_norm.item()
                # ---- End C005b modification ----

                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # C005b: Log per-group gradient norms
        self.logger.record("train/actor_grad_norm", self.last_actor_grad_norm)
        self.logger.record("train/critic_grad_norm", self.last_critic_grad_norm)
