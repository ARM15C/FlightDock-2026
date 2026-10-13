#!/usr/bin/env python3
"""FlightDock DRL Evaluation Script.

Loads a trained TD3/SAC/PPO model + VecNormalize stats and runs N episodes
with IDENTICAL termination logic as training.

Supports --baseline-only for pure baseline comparison (zero-action episodes).

Usage:
    # TD3 evaluation (default):
    python src/drl/evaluate.py \\
        --model checkpoints/campaign_006/model_final.zip \\
        --n-episodes 5 \\
        --speedup 1.0 \\
        --output eval_results/campaign_006/

    # PPO evaluation:
    python src/drl/evaluate.py \\
        --model checkpoints/ppo_baseline/model_final.zip \\
        --algorithm ppo \\
        --n-episodes 5 \\
        --output eval_results/ppo_baseline/

    # Baseline-only (no model needed):
    python src/drl/evaluate.py \\
        --baseline-only \\
        --n-episodes 5 \\
        --output eval_results/baseline/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Ensure src/ and tests/ are on path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for d in ("src", "tests"):
    p = os.path.join(_REPO_ROOT, d)
    if p not in sys.path:
        sys.path.insert(0, p)

from stable_baselines3 import TD3, SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from drl import config
from drl.env_formation_sitl import FormationSITLEnv


def _load_model(model_path: str, algorithm: str):
    """Load model using algorithm-specific class."""
    if algorithm == "td3":
        return TD3.load(model_path)
    elif algorithm == "sac":
        return SAC.load(model_path)
    elif algorithm == "ppo":
        return PPO.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def evaluate(
    model_path: str,
    n_episodes: int,
    speedup: float,
    output_dir: str,
    algorithm: str = "td3",
    deterministic: bool = True,
    baseline_only: bool = False,
) -> Dict[str, Any]:
    """Run evaluation episodes and collect metrics.

    Parameters
    ----------
    model_path: str
        Path to the trained model .zip file.
    n_episodes: int
        Number of evaluation episodes.
    speedup: float
        SITL speedup factor.
    output_dir: str
        Directory for output files.
    algorithm: str
        Algorithm type: td3, sac, or ppo.
    deterministic: bool
        Use deterministic actions (default True).
    baseline_only: bool
        Skip model loading; use zero-action [0,0,0] for pure baseline.

    Returns
    -------
    dict
        Summary metrics across all episodes.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model (unless baseline-only)
    model = None
    if not baseline_only:
        logger.info(f"Loading {algorithm.upper()} model from {model_path}")
        model = _load_model(model_path, algorithm)

    # Create env (single instance for eval)
    env = FormationSITLEnv(env_id=0, speedup=speedup)
    vec_env = DummyVecEnv([lambda: env])

    # Load VecNormalize if available (frozen for eval)
    if not baseline_only:
        norm_path = model_path.replace("model_", "vecnormalize_").replace(".zip", ".pkl")
        if not os.path.exists(norm_path):
            norm_path = os.path.join(
                os.path.dirname(model_path),
                "vecnormalize_final.pkl",
            )
        if os.path.exists(norm_path):
            logger.info(f"Loading VecNormalize from {norm_path}")
            vec_env = VecNormalize.load(norm_path, vec_env)
            vec_env.training = False  # Freeze stats for eval
        else:
            logger.warning("No VecNormalize stats found — using raw observations")

    # Run episodes
    episodes: List[Dict[str, Any]] = []

    for ep in range(n_episodes):
        logger.info(f"Episode {ep + 1}/{n_episodes}")
        obs = vec_env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        errors_3d: List[float] = []
        errors_behind: List[float] = []
        errors_cross: List[float] = []
        errors_below: List[float] = []
        ep_actions: List[np.ndarray] = []
        first_m3b_step = None

        # Per-component reward accumulators
        reward_sums: Dict[str, float] = {}

        while not done:
            if baseline_only:
                action = np.zeros((1, config.ACTION_DIM), dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=deterministic)

            obs, reward, dones, infos = vec_env.step(action)
            done = dones[0]
            info = infos[0]
            total_reward += reward[0]
            step_count += 1
            ep_actions.append(action.flatten())

            # Collect error metrics
            if "behind_m" in info:
                behind_err = abs(info.get("behind_m", 0.0))
                cross_err = abs(info.get("cross_m", 0.0))
                below_err = abs(info.get("below_m", 0.0))
                err_3d = math.sqrt(behind_err**2 + cross_err**2 + below_err**2)
                errors_3d.append(err_3d)
                errors_behind.append(behind_err)
                errors_cross.append(cross_err)
                errors_below.append(below_err)

            # Accumulate reward components
            comps = info.get("reward_components", {})
            for key, val in comps.items():
                reward_sums[key] = reward_sums.get(key, 0.0) + float(val)

            if info.get("in_m3b", False) and first_m3b_step is None:
                first_m3b_step = step_count

        # Compute episode metrics (C006 P4: renamed from m3b → precontact_tier4)
        tier4_dwell = info.get("m3b_dwell_s", 0.0)
        terminal_type = info.get("terminal_type", "UNKNOWN")
        tier4_success = tier4_dwell >= config.SUCCESS_DWELL_S

        # Action statistics
        actions_arr = np.array(ep_actions)
        action_l2_norm = float(np.mean(np.linalg.norm(actions_arr, axis=1)))
        action_mean_per_dim = actions_arr.mean(axis=0).tolist()
        action_std_per_dim = actions_arr.std(axis=0).tolist()

        ep_metrics = {
            "episode": ep + 1,
            "total_reward": float(total_reward),
            "steps": step_count,
            "terminal_type": terminal_type,
            "precontact_tier4_success": tier4_success,
            "precontact_tier4_dwell_s": float(tier4_dwell),
            "time_to_first_tier4_steps": first_m3b_step,
            # C006 P4: Actual milestone columns
            "actual_M2_achieved": info.get("actual_M2_achieved", False),
            "actual_M2_dwell_s": info.get("actual_M2_dwell_s", 0.0),
            "actual_M3b_achieved": info.get("actual_M3b_achieved", False),
            "actual_M3b_dwell_s": info.get("actual_M3b_dwell_s", 0.0),
            # Action statistics
            "mean_action_l2_norm": action_l2_norm,
            "mean_action_speed": (
                float(action_mean_per_dim[0])
                if len(action_mean_per_dim) > 0 else 0.0),
            "mean_action_heading": (
                float(action_mean_per_dim[1])
                if len(action_mean_per_dim) > 1 else 0.0),
            "mean_action_altitude": (
                float(action_mean_per_dim[2])
                if len(action_mean_per_dim) > 2 else 0.0),
            "std_action_speed": (
                float(action_std_per_dim[0])
                if len(action_std_per_dim) > 0 else 0.0),
            "std_action_heading": (
                float(action_std_per_dim[1])
                if len(action_std_per_dim) > 1 else 0.0),
            "std_action_altitude": (
                float(action_std_per_dim[2])
                if len(action_std_per_dim) > 2 else 0.0),
        }

        # Per-component reward means
        if step_count > 0:
            for key in ["r_ref", "r_prox", "r_dwell", "r_safe", "r_smooth",
                        "r_wake", "r_m3a_coarse", "r_m3a_fine"]:
                ep_metrics[f"{key}_mean"] = reward_sums.get(key, 0.0) / step_count

        if errors_3d:
            arr_3d = np.array(errors_3d)
            arr_b = np.array(errors_behind)
            arr_c = np.array(errors_cross)
            arr_bl = np.array(errors_below)
            ep_metrics.update({
                "rms_3d_m": float(np.sqrt(np.mean(arr_3d**2))),
                "rms_behind_m": float(np.sqrt(np.mean(arr_b**2))),
                "rms_cross_m": float(np.sqrt(np.mean(arr_c**2))),
                "rms_below_m": float(np.sqrt(np.mean(arr_bl**2))),
                "p95_3d_m": float(np.percentile(arr_3d, 95)),
                "peak_3d_m": float(np.max(arr_3d)),
            })

        episodes.append(ep_metrics)
        logger.info(
            f"  reward={total_reward:.1f}  dwell={tier4_dwell:.1f}s  "
            f"terminal={terminal_type}  tier4={'YES' if tier4_success else 'NO'}  "
            f"action_l2={action_l2_norm:.4f}"
        )

    vec_env.close()

    # Summary
    # C006 P4: Renamed from m3b → precontact_tier4
    n_success = sum(1 for e in episodes if e["precontact_tier4_success"])
    summary = {
        "model_path": model_path if not baseline_only else "BASELINE_ONLY",
        "algorithm": algorithm,
        "n_episodes": n_episodes,
        "speedup": speedup,
        "deterministic": deterministic,
        "baseline_only": baseline_only,
        "precontact_tier4_success_rate": n_success / max(n_episodes, 1),
        "mean_reward": float(np.mean([e["total_reward"] for e in episodes])),
        "mean_precontact_tier4_dwell_s": float(np.mean(
            [e["precontact_tier4_dwell_s"] for e in episodes])),
        "mean_action_l2_norm": float(np.mean([e["mean_action_l2_norm"] for e in episodes])),
        "episodes": episodes,
    }

    # Write JSON
    json_path = os.path.join(output_dir, "eval_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results written to {json_path}")

    # Write CSV
    csv_path = os.path.join(output_dir, "eval_results.csv")
    if episodes:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=episodes[0].keys())
            writer.writeheader()
            for ep in episodes:
                writer.writerow(ep)
        logger.info(f"CSV written to {csv_path}")

    # Write Markdown summary
    md_path = os.path.join(output_dir, "eval_summary.md")
    with open(md_path, "w") as f:
        f.write("# DRL Evaluation Summary\n\n")
        f.write(f"- Model: `{summary['model_path']}`\n")
        f.write(f"- Algorithm: {algorithm.upper()}\n")
        f.write(f"- Episodes: {n_episodes}\n")
        f.write(f"- Speedup: {speedup}x\n")
        f.write(f"- Deterministic: {deterministic}\n")
        f.write(f"- Baseline only: {baseline_only}\n")
        f.write(
            f"- Precontact tier 4 success rate: {summary['precontact_tier4_success_rate']:.1%}\n")
        f.write(f"- Mean reward: {summary['mean_reward']:.1f}\n")
        f.write(f"- Mean tier 4 dwell: {summary['mean_precontact_tier4_dwell_s']:.1f}s\n")
        f.write(f"- Mean action L2 norm: {summary['mean_action_l2_norm']:.4f}\n\n")
        f.write("## Per-Episode Results\n\n")
        f.write("| Ep | Reward | Dwell | Terminal | Tier4 | Action L2 |\n")
        f.write("|:--:|-------:|------:|---------:|:-----:|----------:|\n")
        for e in episodes:
            t4_str = "YES" if e["precontact_tier4_success"] else "NO"
            f.write(
                f"| {e['episode']} | {e['total_reward']:.1f} | "
                f"{e['precontact_tier4_dwell_s']:.1f}s | {e['terminal_type']} | {t4_str} | "
                f"{e['mean_action_l2_norm']:.4f} |\n"
            )
    logger.info(f"Summary written to {md_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlightDock DRL Evaluation (TD3/SAC/PPO)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model .zip file",
    )
    parser.add_argument(
        "--algorithm", type=str, default="td3",
        choices=["td3", "sac", "ppo"],
        help="Algorithm type (default: td3)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=5,
        help="Number of evaluation episodes (default: 5)",
    )
    parser.add_argument(
        "--speedup", type=float, default=1.0,
        help="SITL speedup factor (default: 1.0)",
    )
    parser.add_argument(
        "--output", type=str, default="eval_results/default",
        help="Output directory for results",
    )
    parser.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic actions (default: True)",
    )
    parser.add_argument(
        "--no-deterministic", action="store_true", default=False,
        help="Use stochastic actions (overrides --deterministic)",
    )
    parser.add_argument(
        "--baseline-only", action="store_true", default=False,
        help="Skip model loading; use zero-action [0,0,0] for pure baseline comparison",
    )
    args = parser.parse_args()

    if not args.baseline_only and args.model is None:
        parser.error("--model is required unless --baseline-only is set")

    deterministic = not args.no_deterministic

    evaluate(
        model_path=args.model or "",
        n_episodes=args.n_episodes,
        speedup=args.speedup,
        output_dir=args.output,
        algorithm=args.algorithm,
        deterministic=deterministic,
        baseline_only=args.baseline_only,
    )


if __name__ == "__main__":
    main()
