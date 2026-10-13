"""Demo-augmented replay buffer for behaviour cloning hybrid (v19 P7).

Provides ``DemoReplayBuffer``, a drop-in replacement for SB3's
``ReplayBuffer`` that mixes pre-collected demonstration transitions with
online experience. When ``demo_ratio=0`` or no demos are loaded, the
buffer behaves identically to a standard ReplayBuffer.

This is an **escalation tool** — dormant in normal training. Activated
only if TD3+SAC both fail to escape zero-residual, via ``collect_pid_demos.py``
→ pickle → ``DemoReplayBuffer(demo_ratio=0.25)``.

Usage:
    from drl.demo_buffer import DemoReplayBuffer

    buffer = DemoReplayBuffer(
        buffer_size=50_000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        demo_ratio=0.25, # 25% of each batch from demos
    )
    buffer.load_demos("demos/pid_baseline.pkl")
"""

from __future__ import annotations

import logging
import pickle
from typing import Optional, Union

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {"observations", "next_observations", "actions", "rewards", "dones"}


class DemoReplayBuffer(ReplayBuffer):
    """ReplayBuffer with demonstration mixing.

    Overrides ``sample()`` to return a batch composed of:
      - ``demo_ratio`` fraction from pre-loaded demos (uniform random)
      - ``(1 - demo_ratio)`` fraction from online experience (standard sampling)

    Parameters
    ----------
    buffer_size: int
        Online buffer capacity (same as ReplayBuffer).
    observation_space: spaces.Space
        Observation space.
    action_space: spaces.Space
        Action space.
    demo_ratio: float
        Fraction of each sampled batch drawn from demonstrations.
        0.0 = pure online, 1.0 = pure demonstration (not recommended).
    device: str
        Torch device for samples.
    n_envs: int
        Number of parallel environments.
    optimize_memory_usage: bool
        SB3 memory optimization flag.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        demo_ratio: float = 0.0,
        device: Union[str, object] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ) -> None:
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
        )
        self.demo_ratio = demo_ratio
        self._demo_obs: Optional[np.ndarray] = None
        self._demo_next_obs: Optional[np.ndarray] = None
        self._demo_actions: Optional[np.ndarray] = None
        self._demo_rewards: Optional[np.ndarray] = None
        self._demo_dones: Optional[np.ndarray] = None
        self._demo_size: int = 0

    @property
    def demo_size(self) -> int:
        """Number of demonstration transitions loaded."""
        return self._demo_size

    def load_demos(self, path: str) -> None:
        """Load demonstration transitions from a pickle file.

        Expected format: dict with keys
        ``{observations, next_observations, actions, rewards, dones}``,
        each a numpy array with matching first dimension.

        Raises
        ------
        ValueError
            If required keys are missing or array shapes are inconsistent.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        missing = _REQUIRED_KEYS - set(data.keys())
        if missing:
            raise ValueError(f"Demo pickle missing keys: {missing}")

        n = len(data["observations"])
        for key in _REQUIRED_KEYS:
            if len(data[key]) != n:
                raise ValueError(
                    f"Demo array '{key}' has length {len(data[key])}, "
                    f"expected {n} (matching 'observations')"
                )

        self._demo_obs = np.asarray(data["observations"], dtype=np.float32)
        self._demo_next_obs = np.asarray(data["next_observations"], dtype=np.float32)
        self._demo_actions = np.asarray(data["actions"], dtype=np.float32)
        self._demo_rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1, 1)
        self._demo_dones = np.asarray(data["dones"], dtype=np.float32).reshape(-1, 1)
        self._demo_size = n
        logger.info(f"[DemoBuffer] Loaded {n} demo transitions from {path}")

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        """Sample a mixed batch of demos + online experience.

        If no demos loaded or demo_ratio=0, falls back to standard sampling.
        """
        if self._demo_size == 0 or self.demo_ratio <= 0.0:
            return super().sample(batch_size, env=env)

        # Compute split
        n_demo = min(int(batch_size * self.demo_ratio), self._demo_size)
        n_online = batch_size - n_demo

        if n_online <= 0 or self.size() == 0:
            # Edge case: all demos or empty online buffer
            return super().sample(batch_size, env=env)

        # Sample online portion via parent
        online_samples = super().sample(n_online, env=env)

        # Sample demo portion (uniform random)
        demo_idx = np.random.randint(0, self._demo_size, size=n_demo)

        # Concatenate — convert demo arrays to tensors matching online format
        import torch

        def _cat(online_tensor, demo_array):
            demo_tensor = torch.as_tensor(demo_array, device=online_tensor.device)
            return torch.cat([online_tensor, demo_tensor], dim=0)

        return ReplayBufferSamples(
            observations=_cat(online_samples.observations, self._demo_obs[demo_idx]),
            actions=_cat(online_samples.actions, self._demo_actions[demo_idx]),
            next_observations=_cat(online_samples.next_observations,
                                   self._demo_next_obs[demo_idx]),
            dones=_cat(online_samples.dones, self._demo_dones[demo_idx]),
            rewards=_cat(online_samples.rewards, self._demo_rewards[demo_idx]),
        )
