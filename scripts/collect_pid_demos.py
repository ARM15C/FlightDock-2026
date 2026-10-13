#!/usr/bin/env python3
"""Collect PID baseline demonstration data for demo injection (v19 P7).

Runs the baseline controller (action=[0,0,0] = pure MRRL baseline passthrough)
through SITL and records all transitions for later injection into the replay
buffer via DemoReplayBuffer.

This script requires a running SITL instance — it is NOT run during unit tests.

Usage:
    python scripts/collect_pid_demos.py \\
        --n-episodes 30 \\
        --output demos/pid_baseline.pkl \\
        --speedup 2.0

Output format: pickle dict with keys:
    {observations, next_observations, actions, rewards, dones}
Each value is a numpy array. observations/next_observations have shape (N, 39),
actions (N, 3), rewards (N,), dones (N,).
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

# Ensure src/ is on path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("src", "tests"):
    p = os.path.join(_REPO_ROOT, d)
    if p not in sys.path:
        sys.path.insert(0, p)


def collect_demos(
    n_episodes: int,
    speedup: float,
    output_path: str,
    ui_mode: str = "none",
) -> None:
    """Collect demonstration transitions from baseline controller."""
    from drl import config
    from drl.curriculum import CurriculumManager
    from drl.env_formation_sitl import FormationSITLEnv

    # Use Stage 0 settings (no wind)
    curriculum = CurriculumManager(initial_stage=0)

    env = FormationSITLEnv(
        env_id=0,
        speedup=speedup,
        curriculum_manager=curriculum,
        ui_mode=ui_mode,
    )

    all_obs = []
    all_next_obs = []
    all_actions = []
    all_rewards = []
    all_dones = []

    try:
        for ep in range(n_episodes):
            obs, info = env.reset()
            done = False
            step_count = 0

            while not done:
                # Baseline passthrough: zero residual action
                action = np.zeros(config.ACTION_DIM, dtype=np.float32)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                all_obs.append(obs.copy())
                all_next_obs.append(next_obs.copy())
                all_actions.append(action.copy())
                all_rewards.append(float(reward))
                all_dones.append(done)

                obs = next_obs
                step_count += 1

            print(f"Episode {ep+1}/{n_episodes}: {step_count} steps, "
                  f"reward={sum(all_rewards[-step_count:]):.2f}")

    finally:
        env.close()

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    data = {
        "observations": np.array(all_obs, dtype=np.float32),
        "next_observations": np.array(all_next_obs, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.float32),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "dones": np.array(all_dones, dtype=np.float32),
    }
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved {len(all_obs)} transitions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect PID baseline demonstrations for demo injection",
    )
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--output", type=str, default="demos/pid_baseline.pkl")
    parser.add_argument("--speedup", type=float, default=2.0)
    parser.add_argument("--ui", type=str, default="none",
                        choices=["none", "map", "console", "both"])
    args = parser.parse_args()

    collect_demos(
        n_episodes=args.n_episodes,
        speedup=args.speedup,
        output_path=args.output,
        ui_mode=args.ui,
    )


if __name__ == "__main__":
    main()
