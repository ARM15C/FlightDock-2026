#!/usr/bin/env python3
"""DRL Post-Campaign Analysis — 6-section analysis with decision output.

Reads metrics.csv and diagnostics CSV from a completed (or in-progress)
campaign, produces a markdown report with analysis sections, and prints a
PROCEED/EXTEND/ITERATE/FIX recommendation to stdout.

Supports TD3/SAC (off-policy) and PPO (on-policy) campaigns via --algorithm flag.
Off-policy campaigns read offpolicy_diagnostics.csv; PPO reads policy_diagnostics.csv.

Usage:
    python scripts/drl_campaign_analysis.py --log-dir logs/campaign_006
    python scripts/drl_campaign_analysis.py --log-dir logs/campaign_004 --algorithm ppo

Output:
    {log-dir}/campaign_analysis.md — full markdown report
    stdout — decision line (PROCEED/EXTEND/ITERATE/FIX)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.drl.parse_helpers import coerce_achievement_series  # noqa: E402

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("ERROR: numpy and pandas are required. Install: pip install numpy pandas",
          file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Thresholds for decision logic
# ---------------------------------------------------------------------------
# These thresholds define what "good" looks like for each metric.
# Derived from the v5/v6/v7 fix validation results (Gate A/B).
# Updated v14: LOG_STD_INIT changed from -1.5 to -0.7, vf_coef 0.5→0.25.

THRESHOLDS = {
    # Section 1: SIGSTOP health
    "max_stopped_wall_s": 300.0,      # Warn if any episode stopped > 5min
    # Section 2: Reward trend
    "r_prox_improvement_pct": 5.0,    # Need >= 5% improvement over campaign
    "safety_range_rate_max": 0.10,    # Fail if > 10% SAFETY_RANGE episodes
    # Section 3: Policy stability
    "approx_kl_max": 0.02,           # Warn if KL exceeds this
    "log_std_drift_max": 0.5,        # Warn if log_std drifts > 0.5 from init (-0.7)
    "gradient_norm_max": 10.0,       # Warn if gradient norm > 10
    # Section 4: Milestone progression
    "min_tier_median": 1.0,          # Need median tier >= M2
    "m3b_dwell_target_s": 1.0,       # Good if any episode achieves this
    # Section 5: Episode health
    "min_episodes": 4,               # Need at least this many episodes
    "timeout_rate_min": 0.5,         # At least 50% TIMEOUT (not SAFETY_RANGE)
}


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score 95% CI for a binomial proportion. Returns (p_hat, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return p, max(0.0, c - m), min(1.0, c + m)


# ---------------------------------------------------------------------------
# Escalation trigger procedure (v14)
# ---------------------------------------------------------------------------
# Evaluated manually at rollout checkpoints during Campaign C005a/C005b.
#
# R2 ABORT (rollout 2 — early gradient health check):
#   IF actor_grad_norm < 0.01 (from policy_diagnostics.csv)
#   THEN: ABORT campaign. Separate clipping is not working at this P value.
#   This catches catastrophic gradient monopoly before wasting 6+ hours.
#
# FIRST ESCALATION (rollout 10):
#   IF median(mean_abs_action_speed) < 0.10 (from policy_diagnostics.csv)
#   AND pv_ratio < 0.02 (still value-dominated)
#   AND log_std has not increased from -0.7
#   THEN: ent_coef 0.02 → 0.04
#
# SECOND ESCALATION (rollout 15, only if first was triggered):
#   IF same conditions persist
#   THEN: also action_scale 0.5 → 0.8
#
# Rationale: At v13, LOG_STD_INIT=-1.5 had negative entropy (-0.24 nats),
# so ent_coef acted as exploration *penalty*. v14 fixes this (LOG_STD_INIT=-0.7,
# entropy +0.42 nats), but if the agent still collapses exploration, doubling
# ent_coef provides a stronger exploration incentive. The second escalation
# widens the action scale to allow larger deviations from baseline.


# ---------------------------------------------------------------------------
# CSV loading with backward compatibility
# ---------------------------------------------------------------------------

def load_metrics(log_dir: Path) -> Optional[pd.DataFrame]:
    """Load metrics.csv with backward-compatible column handling."""
    path = log_dir / "metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def load_policy_diagnostics(log_dir: Path) -> Optional[pd.DataFrame]:
    """Load policy_diagnostics.csv, filtering NaN-only rollout 1."""
    path = log_dir / "policy_diagnostics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def load_offpolicy_diagnostics(log_dir: Path) -> Optional[pd.DataFrame]:
    """Load offpolicy_diagnostics.csv (TD3/SAC campaigns)."""
    path = log_dir / "offpolicy_diagnostics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


# ---------------------------------------------------------------------------
# Analysis sections
# ---------------------------------------------------------------------------

def analyze_sigstop(df: pd.DataFrame) -> Dict[str, Any]:
    """Section 1: SIGSTOP/SIGCONT health metrics.

    C006 B4: Threshold scales by n_envs because DummyVecEnv serial stepping
    accumulates SIGSTOP wall time across all environments. At P=4, Gate G
    showed 543s mean stopped time — exceeds the base 300s threshold despite
    0 STALL_SIMTIME violations. STALL_SIMTIME is the real safety indicator,
    not stopped wall time.
    """
    result: Dict[str, Any] = {"available": False}
    if "stopped_total_wall_s" not in df.columns:
        result["note"] = "SIGSTOP columns not present (pre-v11 format)"
        return result

    result["available"] = True
    result["mean_stopped_wall_s"] = float(df["stopped_total_wall_s"].mean())
    result["max_stopped_wall_s"] = float(df["stopped_total_wall_s"].max())
    result["mean_pause_count"] = float(df["pause_count"].mean()) if "pause_count" in df.columns else 0.0

    # C006 B4: P-aware threshold — scale by n_envs for DummyVecEnv serial stepping
    n_envs = max(int(df["env_id"].dropna().nunique()), 1) if "env_id" in df.columns else 1
    scaled_threshold = THRESHOLDS["max_stopped_wall_s"] * n_envs
    result["healthy"] = result["max_stopped_wall_s"] < scaled_threshold
    result["n_envs_inferred"] = n_envs
    result["scaled_threshold_s"] = scaled_threshold

    # v12: STALL_SIMTIME tracking (true safety indicator, not stopped wall time)
    if "terminal_type" in df.columns:
        stall_count = int((df["terminal_type"] == "STALL_SIMTIME").sum())
        result["stall_simtime_count"] = stall_count
    else:
        result["stall_simtime_count"] = 0

    return result


def analyze_reward_trend(df: pd.DataFrame) -> Dict[str, Any]:
    """Section 2: Reward component trends."""
    result: Dict[str, Any] = {}
    n = len(df)
    result["n_episodes"] = n

    # Overall reward stats
    result["reward_mean"] = float(df["reward"].mean())
    result["reward_std"] = float(df["reward"].std()) if n > 1 else 0.0

    # r_prox trend (rolling 4-episode window)
    # C006 B4: Use gradient-active episodes only when available (pre-learning_starts
    # episodes use random noise, not learned policy — they dilute the trend signal)
    trend_df = df
    if "is_gradient_active" in df.columns:
        grad_active = df[df["is_gradient_active"] == True]  # noqa: E712
        if len(grad_active) >= 4:
            trend_df = grad_active
            result["r_prox_gradient_active_only"] = True
        else:
            result["r_prox_gradient_active_only"] = False
    else:
        result["r_prox_gradient_active_only"] = False

    if "r_prox" in trend_df.columns and len(trend_df) >= 4:
        r_prox = trend_df["r_prox"].values
        first_window = float(np.mean(r_prox[:4]))
        last_window = float(np.mean(r_prox[-4:]))
        result["r_prox_first4"] = first_window
        result["r_prox_last4"] = last_window
        # Improvement: r_prox is negative, so improvement means less negative
        if abs(first_window) > 1e-6:
            result["r_prox_improvement_pct"] = ((last_window - first_window) / abs(first_window)) * 100.0
        else:
            result["r_prox_improvement_pct"] = 0.0
    else:
        result["r_prox_improvement_pct"] = 0.0

    # Terminal type distribution
    if "terminal_type" in df.columns:
        counts = df["terminal_type"].value_counts()
        result["terminal_distribution"] = counts.to_dict()
        safety_count = counts.get("SAFETY_RANGE", 0)
        result["safety_range_rate"] = safety_count / max(n, 1)
    else:
        result["terminal_distribution"] = {}
        result["safety_range_rate"] = 0.0

    # Per-component means (v13: prefer _mean columns when present)
    for col in ["r_ref", "r_prox", "r_dwell", "r_safe", "r_smooth", "r_wake",
                "r_m3a_coarse", "r_m3a_fine",
                "r_terminal", "r_range_penalty"]:
        mean_col = f"{col}_mean"
        source = mean_col if mean_col in df.columns else col
        if source in df.columns:
            result[f"mean_{col}"] = float(df[source].mean())

    # v12: Band reward activation counts (v13: prefer _mean columns)
    for band_col in ["r_m3a_coarse", "r_m3a_fine"]:
        source = f"{band_col}_mean" if f"{band_col}_mean" in df.columns else band_col
        if source in df.columns:
            result[f"{band_col}_active_count"] = int((df[source] > 0).sum())
        else:
            result[f"{band_col}_active_count"] = 0

    return result


def analyze_offpolicy_stability(diag_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Section 3 (off-policy): Policy stability from offpolicy_diagnostics.csv."""
    result: Dict[str, Any] = {"available": False, "offpolicy": True}
    if diag_df is None or diag_df.empty:
        result["note"] = "No offpolicy_diagnostics.csv found"
        return result

    result["available"] = True
    result["n_episodes"] = len(diag_df)

    # action_l2_norm trend
    if "action_l2_norm" in diag_df.columns:
        l2 = diag_df["action_l2_norm"].dropna()
        if len(l2) > 0:
            result["action_l2_mean"] = float(l2.mean())
            result["action_l2_last"] = float(l2.iloc[-1])
            result["action_l2_first"] = float(l2.iloc[0])
            # Learning signal: action magnitude should increase from near-zero
            result["learning_signal"] = result["action_l2_last"] > 0.05

    # actor/critic loss
    if "actor_loss" in diag_df.columns:
        vals = diag_df["actor_loss"].dropna()
        if len(vals) > 0:
            result["actor_loss_mean"] = float(vals.mean())
            result["actor_loss_last"] = float(vals.iloc[-1])

    if "critic_loss" in diag_df.columns:
        vals = diag_df["critic_loss"].dropna()
        if len(vals) > 0:
            result["critic_loss_mean"] = float(vals.mean())
            result["critic_loss_last"] = float(vals.iloc[-1])

    # Buffer utilisation
    if "buffer_utilisation" in diag_df.columns:
        vals = diag_df["buffer_utilisation"].dropna()
        if len(vals) > 0:
            result["buffer_util_last"] = float(vals.iloc[-1])

    # Alert counts
    for alert in ["zero_residual_alert", "q_divergence_alert", "dimension_collapse_alert"]:
        if alert in diag_df.columns:
            # Handle both string and bool columns
            alert_vals = diag_df[alert]
            count = int(alert_vals.apply(
                lambda x: str(x).lower() in ("true", "1", "yes")
            ).sum())
            result[alert + "_count"] = count

    # R2 abort: action_l2_norm < 0.05 at episode >= 10
    if "action_l2_norm" in diag_df.columns and "episode" in diag_df.columns:
        r2_rows = diag_df[diag_df["episode"] >= 10]
        if len(r2_rows) > 0:
            last_l2 = r2_rows["action_l2_norm"].dropna()
            if len(last_l2) > 0 and float(last_l2.iloc[-1]) < 0.05:
                result["r2_abort"] = True
                print("[CRITICAL] R2 ABORT: action_l2_norm < 0.05 at episode >= 10. "
                      "TD3/SAC is not exploring.", file=sys.stderr)
            else:
                result["r2_abort"] = False
        else:
            result["r2_abort"] = False

    # Per-axis action magnitude
    for axis in ["speed", "heading", "altitude"]:
        col = f"action_mean_{axis}"
        if col in diag_df.columns:
            vals = diag_df[col].dropna()
            if len(vals) > 0:
                result[f"action_{axis}_mean"] = float(vals.mean())
                result[f"action_{axis}_last"] = float(vals.iloc[-1])

    return result


def analyze_policy_stability(diag_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Section 3 (PPO): Policy stability from policy_diagnostics.csv."""
    result: Dict[str, Any] = {"available": False, "offpolicy": False}
    if diag_df is None or diag_df.empty:
        result["note"] = "No policy_diagnostics.csv found"
        return result

    result["available"] = True
    result["n_rollouts"] = len(diag_df)

    # Filter out rollout 1 for train/* metrics (SB3 calls _on_rollout_end
    # BEFORE first train(), so rollout 1 has NaN for approx_kl, clip_fraction,
    # explained_variance, entropy_loss, value_loss, gradient_norm).
    # log_std and action magnitudes ARE valid for rollout 1 (policy state, not
    # train output), so those use unfiltered diag_df below.
    valid = diag_df[diag_df["rollout"] > 1] if len(diag_df) > 1 else diag_df

    # approx_kl (train metric — use filtered)
    if "approx_kl" in valid.columns:
        kl_vals = valid["approx_kl"].dropna()
        if len(kl_vals) > 0:
            result["approx_kl_mean"] = float(kl_vals.mean())
            result["approx_kl_max"] = float(kl_vals.max())
            result["kl_healthy"] = result["approx_kl_max"] < THRESHOLDS["approx_kl_max"]

    # log_std drift
    if "log_std_mean" in diag_df.columns:
        log_std = diag_df["log_std_mean"].dropna()
        if len(log_std) > 0:
            result["log_std_first"] = float(log_std.iloc[0])
            result["log_std_last"] = float(log_std.iloc[-1])
            result["log_std_drift"] = abs(result["log_std_last"] - result["log_std_first"])
            result["log_std_healthy"] = result["log_std_drift"] < THRESHOLDS["log_std_drift_max"]

    # gradient norm
    if "gradient_norm" in valid.columns:
        grad = valid["gradient_norm"].dropna()
        if len(grad) > 0:
            result["gradient_norm_mean"] = float(grad.mean())
            result["gradient_norm_max"] = float(grad.max())
            result["gradient_healthy"] = result["gradient_norm_max"] < THRESHOLDS["gradient_norm_max"]

    # actor_grad_norm (C005b separate clipping — R2 abort check)
    if "actor_grad_norm" in valid.columns:
        agn = valid["actor_grad_norm"].dropna()
        if len(agn) > 0:
            result["actor_grad_norm_mean"] = float(agn.mean())
            result["actor_grad_norm_last"] = float(agn.iloc[-1])
            # R2 ABORT trigger: if rollout >= 2 and actor_grad_norm < 0.01,
            # separate clipping is not working — gradient monopoly persists.
            r2_rows = valid[valid["rollout"] >= 2] if "rollout" in valid.columns else valid
            if len(r2_rows) > 0:
                r2_agn = r2_rows["actor_grad_norm"].dropna()
                if len(r2_agn) > 0 and float(r2_agn.iloc[-1]) < 0.01:
                    result["r2_abort"] = True
                    print("[CRITICAL] R2 ABORT: actor_grad_norm < 0.01 at rollout >= 2. "
                          "Separate clipping may not be working.", file=sys.stderr)
                else:
                    result["r2_abort"] = False
            else:
                result["r2_abort"] = False

    # Per-axis action magnitude
    for axis in ["speed", "heading", "altitude"]:
        col = f"mean_abs_action_{axis}"
        if col in diag_df.columns:
            vals = diag_df[col].dropna()
            if len(vals) > 0:
                result[f"action_{axis}_mean"] = float(vals.mean())
                result[f"action_{axis}_last"] = float(vals.iloc[-1])

    return result


def analyze_milestones(df: pd.DataFrame) -> Dict[str, Any]:
    """Section 4: Milestone tier progression.

    C006 P4: CSV columns renamed from best_milestone_tier → best_precontact_tier,
    dwell_M3b_s → dwell_precontact_tier4_s, etc. Backward-compatible with old names.
    """
    result: Dict[str, Any] = {"available": False}

    # C006 P4: Support both old and new column names
    tier_col = (
        "best_precontact_tier" if "best_precontact_tier" in df.columns
        else "best_milestone_tier" if "best_milestone_tier" in df.columns
        else None
    )
    if tier_col is None:
        result["note"] = "Milestone columns not present (pre-v11 format)"
        return result

    result["available"] = True
    tiers = df[tier_col]
    result["tier_median"] = float(tiers.median())
    result["tier_max"] = int(tiers.max())
    result["tier_distribution"] = tiers.value_counts().sort_index().to_dict()

    # Tier 4 (M3b) dwell stats — support old and new column names
    dwell_col = (
        "dwell_precontact_tier4_s" if "dwell_precontact_tier4_s" in df.columns
        else "dwell_M3b_s" if "dwell_M3b_s" in df.columns
        else None
    )
    if dwell_col is not None:
        dwell = df[dwell_col]
        result["tier4_dwell_max"] = float(dwell.max())
        result["tier4_dwell_mean"] = float(dwell.mean())
        result["tier4_episodes"] = int((dwell > 0).sum())

    # Time-to-milestone stats — C006 P4: precontact tier naming
    tier_names = {1: "tier1", 2: "tier2", 3: "tier3", 4: "tier4"}
    old_names = {1: "M2", 2: "M3a_coarse", 3: "M3a_fine", 4: "M3b"}
    for tier_num, tier_name in tier_names.items():
        new_col = f"time_to_precontact_{tier_name}_s"
        old_col = f"time_to_{old_names[tier_num]}_s"
        col = new_col if new_col in df.columns else old_col if old_col in df.columns else None
        if col is not None:
            vals = df[col]
            reached = vals[vals >= 0]
            if len(reached) > 0:
                result[f"time_to_{tier_name}_mean"] = float(reached.mean())
                result[f"{tier_name}_reach_rate"] = len(reached) / len(vals)

    # C006 P4: Actual milestone reporting
    if "actual_M2_achieved" in df.columns:
        result["actual_M2_rate"] = float(coerce_achievement_series(df["actual_M2_achieved"]).mean())
    if "actual_M3b_achieved" in df.columns:
        result["actual_M3b_rate"] = float(coerce_achievement_series(df["actual_M3b_achieved"]).mean())
    if "actual_M3b_dwell_s" in df.columns:
        result["actual_M3b_dwell_max"] = float(df["actual_M3b_dwell_s"].max())

    return result


def analyze_episode_health(df: pd.DataFrame) -> Dict[str, Any]:
    """Section 5: Episode-level health metrics."""
    result: Dict[str, Any] = {}
    n = len(df)
    result["n_episodes"] = n

    if "length" in df.columns:
        result["length_mean"] = float(df["length"].mean())
        result["length_std"] = float(df["length"].std()) if n > 1 else 0.0
        result["length_min"] = int(df["length"].min())
        result["length_max"] = int(df["length"].max())

    if "env_id" in df.columns:
        result["envs_seen"] = sorted(df["env_id"].dropna().unique().tolist())
    else:
        result["envs_seen"] = []

    if "min_range_m" in df.columns:
        result["min_range_mean"] = float(df["min_range_m"].mean())
        result["min_range_best"] = float(df["min_range_m"].min())

    return result


def analyze_learned_vs_baseline(df: pd.DataFrame) -> Dict[str, Any]:
    """Section 6: Learned-vs-baseline residual analysis (v12).

    C006 P4: CSV columns renamed from m3b_mean_residual → precontact_tier4_mean_residual,
    learned_m3b_fraction → learned_precontact_tier4_fraction, etc.
    Backward-compatible with old column names.
    """
    result: Dict[str, Any] = {"available": False}

    # C006 P4: Support both old and new column names
    residual_col = (
        "precontact_tier4_mean_residual" if "precontact_tier4_mean_residual" in df.columns
        else "m3b_mean_residual" if "m3b_mean_residual" in df.columns
        else None
    )
    fraction_col = (
        "learned_precontact_tier4_fraction" if "learned_precontact_tier4_fraction" in df.columns
        else "learned_m3b_fraction" if "learned_m3b_fraction" in df.columns
        else None
    )
    if residual_col is None or fraction_col is None:
        result["note"] = "Residual/fraction columns not found (pre-v12 data)"
        return result
    if "residual_prox_correlation" not in df.columns:
        result["note"] = "residual_prox_correlation not found (pre-v12 data)"
        return result

    result["available"] = True
    result["tier4_mean_residual_overall"] = float(df[residual_col].mean())
    result["learned_tier4_fraction_overall"] = float(df[fraction_col].mean())

    # Filter out NaN for correlation
    corr_vals = df["residual_prox_correlation"].dropna()
    if len(corr_vals) > 0:
        result["residual_prox_correlation_mean"] = float(corr_vals.mean())
    else:
        result["residual_prox_correlation_mean"] = float("nan")

    # Tier 3 (M3a fine) residual
    tier3_col = (
        "precontact_tier3_mean_residual" if "precontact_tier3_mean_residual" in df.columns
        else "m3a_fine_mean_residual" if "m3a_fine_mean_residual" in df.columns
        else None
    )
    if tier3_col is not None:
        result["tier3_mean_residual_overall"] = float(df[tier3_col].mean())

    # Flag episodes with baseline passthrough
    passthrough_eps = int((df[fraction_col] < 0.10).sum())
    result["baseline_passthrough_episodes"] = passthrough_eps
    result["baseline_passthrough_rate"] = passthrough_eps / max(len(df), 1)

    return result


def check_actor_loss_acceleration(log_dir: str) -> Dict[str, Any]:
    """C006 P9: Detect actor loss divergence via 2nd derivative.

    Reads offpolicy_diagnostics.csv, computes:
    - 1st derivative: episode-over-episode change in actor_loss
    - 2nd derivative: change in growth rate
    Flags WARNING if 2nd derivative is positive for 3+ consecutive episodes,
    indicating accelerating loss growth (potential Q-value explosion).
    """
    result: Dict[str, Any] = {"available": False}
    diag_path = os.path.join(log_dir, "offpolicy_diagnostics.csv")
    if not os.path.exists(diag_path):
        result["note"] = "offpolicy_diagnostics.csv not found"
        return result

    try:
        df = pd.read_csv(diag_path)
    except Exception as e:
        result["note"] = f"Error reading diagnostics: {e}"
        return result

    if "actor_loss" not in df.columns or len(df) < 5:
        result["note"] = "actor_loss column missing or <5 episodes"
        return result

    result["available"] = True
    actor_loss = df["actor_loss"].dropna().values
    result["actor_loss_final"] = float(actor_loss[-1]) if len(actor_loss) > 0 else 0.0

    # First derivative (delta per episode)
    d1 = np.diff(actor_loss)
    # Second derivative (acceleration)
    d2 = np.diff(d1)

    # Check for 3+ consecutive positive 2nd derivative (accelerating growth)
    max_consecutive_pos = 0
    current_streak = 0
    for val in d2:
        if val > 0:
            current_streak += 1
            max_consecutive_pos = max(max_consecutive_pos, current_streak)
        else:
            current_streak = 0

    result["max_consecutive_pos_d2"] = max_consecutive_pos
    result["actor_loss_diverging"] = max_consecutive_pos >= 3

    # Q-value proxy (critic loss)
    if "critic_loss" in df.columns:
        critic_vals = df["critic_loss"].dropna().values
        if len(critic_vals) > 0:
            result["critic_loss_final"] = float(critic_vals[-1])
            result["critic_loss_mean"] = float(np.mean(critic_vals))

    return result


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def compute_decision(
    reward: Dict[str, Any],
    policy: Dict[str, Any],
    milestones: Dict[str, Any],
    episode: Dict[str, Any],
) -> tuple[str, List[str]]:
    """Compute PROCEED/EXTEND/ITERATE/FIX decision with reasons.

    Decision hierarchy:
    - FIX: Critical issues that need code changes before relaunch
    - ITERATE: Hyperparameter tuning needed, but infrastructure works
    - EXTEND: Training is progressing but needs more timesteps
    - PROCEED: Ready for next campaign stage
    """
    issues: List[str] = []
    warnings: List[str] = []

    n = episode.get("n_episodes", 0)

    # FIX conditions
    if n < THRESHOLDS["min_episodes"]:
        issues.append(f"Too few episodes ({n} < {THRESHOLDS['min_episodes']})")

    safety_rate = reward.get("safety_range_rate", 0.0)
    if safety_rate > THRESHOLDS["safety_range_rate_max"]:
        issues.append(f"SAFETY_RANGE rate {safety_rate:.1%} > {THRESHOLDS['safety_range_rate_max']:.0%}")

    if policy.get("available"):
        if policy.get("offpolicy"):
            # Off-policy: check for R2 abort
            if policy.get("r2_abort", False):
                issues.append("R2 ABORT: action_l2_norm < 0.05 at episode >= 10")
        else:
            kl_max = policy.get("approx_kl_max", 0.0)
            if kl_max > 0.05:  # Hard KL divergence
                issues.append(f"KL divergence {kl_max:.4f} > 0.05 — policy collapsed")

    if issues:
        return "FIX", issues

    # ITERATE conditions
    if policy.get("available"):
        if policy.get("offpolicy"):
            # Off-policy: check alert counts
            for alert in ["zero_residual_alert_count", "q_divergence_alert_count",
                          "dimension_collapse_alert_count"]:
                count = policy.get(alert, 0)
                if count > 0:
                    warnings.append(f"{alert}: {count} episodes")
            if not policy.get("learning_signal", True):
                warnings.append("No learning signal: action_l2_norm < 0.05 at last episode")
        else:
            if not policy.get("kl_healthy", True):
                warnings.append(f"KL above threshold: {policy.get('approx_kl_max', 0):.4f}")
            if not policy.get("log_std_healthy", True):
                warnings.append(f"log_std drift: {policy.get('log_std_drift', 0):.3f}")
            if not policy.get("gradient_healthy", True):
                warnings.append(f"Gradient norm high: {policy.get('gradient_norm_max', 0):.2f}")

    r_prox_impr = reward.get("r_prox_improvement_pct", 0.0)
    if n >= 8 and r_prox_impr < 0:
        warnings.append(f"r_prox worsening: {r_prox_impr:.1f}%")

    if warnings:
        return "ITERATE", warnings

    # EXTEND conditions — training is working but incomplete
    extend_reasons: List[str] = []
    if milestones.get("available"):
        tier_median = milestones.get("tier_median", 0)
        if tier_median < THRESHOLDS["min_tier_median"]:
            extend_reasons.append(f"Median tier {tier_median:.1f} < {THRESHOLDS['min_tier_median']}")
        tier4_dwell = milestones.get("tier4_dwell_max", 0)
        if tier4_dwell < THRESHOLDS["m3b_dwell_target_s"]:
            extend_reasons.append(f"Best tier 4 dwell {tier4_dwell:.1f}s < {THRESHOLDS['m3b_dwell_target_s']}s")

    # Compute actual TIMEOUT rate from terminal distribution (not 1-safety_rate,
    # which conflates SUCCESS with TIMEOUT)
    term_dist = reward.get("terminal_distribution", {})
    timeout_count = term_dist.get("TIMEOUT", 0)
    success_count = term_dist.get("SUCCESS", 0)
    actual_timeout_rate = (timeout_count + success_count) / max(n, 1)
    if actual_timeout_rate < THRESHOLDS["timeout_rate_min"]:
        extend_reasons.append(
            f"TIMEOUT+SUCCESS rate {actual_timeout_rate:.1%} < "
            f"{THRESHOLDS['timeout_rate_min']:.0%} "
            f"(SUCCESS={success_count}, TIMEOUT={timeout_count})"
        )

    if extend_reasons:
        return "EXTEND", extend_reasons

    return "PROCEED", ["All metrics within thresholds"]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    log_dir: Path,
    sigstop: Dict[str, Any],
    reward: Dict[str, Any],
    policy: Dict[str, Any],
    milestones: Dict[str, Any],
    episode: Dict[str, Any],
    decision: str,
    reasons: List[str],
    learned_baseline: Optional[Dict[str, Any]] = None,
    actor_loss_accel: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate the full markdown analysis report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# DRL Campaign Analysis Report",
        f"",
        f"**Generated**: {now}",
        f"**Log directory**: `{log_dir}`",
        f"**Episodes analyzed**: {episode.get('n_episodes', 0)}",
        f"**Decision**: **{decision}**",
        f"",
        f"---",
        f"",
    ]

    # Section 1: SIGSTOP health
    lines.append("## S1: SIGSTOP/SIGCONT Health\n")
    if sigstop.get("available"):
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Mean stopped wall time | {sigstop['mean_stopped_wall_s']:.1f} s |")
        lines.append(f"| Max stopped wall time | {sigstop['max_stopped_wall_s']:.1f} s |")
        lines.append(f"| Mean pause count | {sigstop['mean_pause_count']:.0f} |")
        n_envs = sigstop.get("n_envs_inferred", 1)
        scaled = sigstop.get("scaled_threshold_s", THRESHOLDS["max_stopped_wall_s"])
        lines.append(f"| P-scaled threshold (n_envs={n_envs}) | {scaled:.0f} s |")
        lines.append(f"| Healthy | {'YES' if sigstop['healthy'] else 'NO'} |")
        stall_count = sigstop.get("stall_simtime_count", 0)
        lines.append(f"| STALL_SIMTIME episodes | {stall_count} |")
    else:
        lines.append(f"*{sigstop.get('note', 'Not available')}*")
    lines.append("")

    # Section 2: Reward trend
    lines.append("## S2: Reward Trend\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Mean reward | {reward.get('reward_mean', 0):.3f} |")
    lines.append(f"| Reward std | {reward.get('reward_std', 0):.3f} |")
    if "r_prox_first4" in reward:
        lines.append(f"| r_prox first 4 eps | {reward['r_prox_first4']:.3f} |")
        lines.append(f"| r_prox last 4 eps | {reward['r_prox_last4']:.3f} |")
        lines.append(f"| r_prox improvement | {reward.get('r_prox_improvement_pct', 0):.1f}% |")
    lines.append(f"| SAFETY_RANGE rate | {reward.get('safety_range_rate', 0):.1%} |")
    lines.append("")

    # Terminal distribution
    term_dist = reward.get("terminal_distribution", {})
    if term_dist:
        lines.append("**Terminal type distribution:**\n")
        for ttype, count in sorted(term_dist.items(), key=lambda x: -x[1]):
            lines.append(f"- {ttype}: {count}")
        lines.append("")

    # Component means
    comp_keys = [k for k in reward if k.startswith("mean_r_")]
    if comp_keys:
        lines.append("**Reward component means:**\n")
        lines.append(f"| Component | Mean |")
        lines.append(f"|-----------|------|")
        for k in sorted(comp_keys):
            name = k.replace("mean_", "")
            lines.append(f"| {name} | {reward[k]:.4f} |")
        lines.append("")

    # Section 3: Policy stability
    lines.append("## S3: Policy Stability\n")
    if policy.get("available"):
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        if policy.get("offpolicy"):
            # Off-policy diagnostics
            lines.append(f"| Episodes analyzed | {policy.get('n_episodes', 0)} |")
            if "action_l2_mean" in policy:
                lines.append(f"| Action L2 norm (mean) | {policy['action_l2_mean']:.4f} |")
                lines.append(f"| Action L2 norm (last) | {policy['action_l2_last']:.4f} |")
                ls = "YES" if policy.get("learning_signal", False) else "NO"
                lines.append(f"| Learning signal (L2 > 0.05) | {ls} |")
            if "actor_loss_mean" in policy:
                lines.append(f"| Actor loss (mean) | {policy['actor_loss_mean']:.4f} |")
            if "critic_loss_mean" in policy:
                lines.append(f"| Critic loss (mean) | {policy['critic_loss_mean']:.4f} |")
            if "buffer_util_last" in policy:
                lines.append(f"| Buffer utilisation | {policy['buffer_util_last']:.2%} |")
            for alert in ["zero_residual_alert_count", "q_divergence_alert_count",
                          "dimension_collapse_alert_count"]:
                if alert in policy:
                    lines.append(f"| {alert} | {policy[alert]} |")
        else:
            # PPO diagnostics
            lines.append(f"| Rollouts | {policy.get('n_rollouts', 0)} |")
            if "approx_kl_mean" in policy:
                lines.append(f"| KL mean | {policy['approx_kl_mean']:.6f} |")
                lines.append(f"| KL max | {policy['approx_kl_max']:.6f} |")
            if "log_std_first" in policy:
                lines.append(f"| log_std first | {policy['log_std_first']:.4f} |")
                lines.append(f"| log_std last | {policy['log_std_last']:.4f} |")
                lines.append(f"| log_std drift | {policy.get('log_std_drift', 0):.4f} |")
            if "gradient_norm_mean" in policy:
                lines.append(f"| Gradient norm mean | {policy['gradient_norm_mean']:.4f} |")
                lines.append(f"| Gradient norm max | {policy['gradient_norm_max']:.4f} |")
        for axis in ["speed", "heading", "altitude"]:
            mean_key = f"action_{axis}_mean"
            if mean_key in policy:
                lines.append(f"| Action {axis} mean | {policy[mean_key]:.4f} |")
    else:
        lines.append(f"*{policy.get('note', 'Not available')}*")
    lines.append("")

    # Section 4: Milestone progression
    lines.append("## S4: Milestone Progression\n")
    if milestones.get("available"):
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Tier median | {milestones.get('tier_median', 0):.1f} |")
        lines.append(f"| Tier max | {milestones.get('tier_max', 0)} |")
        if "tier4_dwell_max" in milestones:
            lines.append(f"| Tier 4 dwell max | {milestones['tier4_dwell_max']:.2f} s |")
            lines.append(f"| Tier 4 dwell mean | {milestones['tier4_dwell_mean']:.2f} s |")
            lines.append(f"| Episodes reaching tier 4 | {milestones.get('tier4_episodes', 0)} |")
        # C006 P4: Actual milestone reporting
        if "actual_M3b_rate" in milestones:
            lines.append(f"| Actual M3b rate | {milestones['actual_M3b_rate']:.1%} |")
        if "actual_M3b_dwell_max" in milestones:
            lines.append(f"| Actual M3b dwell max | {milestones['actual_M3b_dwell_max']:.2f} s |")
        if "actual_M2_rate" in milestones:
            lines.append(f"| Actual M2 rate | {milestones['actual_M2_rate']:.1%} |")
        lines.append("")

        # Tier distribution
        tier_dist = milestones.get("tier_distribution", {})
        if tier_dist:
            lines.append("**Tier distribution:**\n")
            tier_names = {0: "NONE", 1: "M2", 2: "M3a_coarse", 3: "M3a_fine", 4: "M3b"}
            for tier, count in sorted(tier_dist.items()):
                name = tier_names.get(int(tier), f"Tier {tier}")
                lines.append(f"- {name} (tier {int(tier)}): {count} episodes")
            lines.append("")

        # Time-to-milestone
        for milestone in ["M2", "M3a_coarse", "M3a_fine", "M3b"]:
            key = f"time_to_{milestone}_mean"
            if key in milestones:
                rate = milestones.get(f"{milestone}_reach_rate", 0)
                lines.append(f"- Time to {milestone}: {milestones[key]:.1f}s mean ({rate:.0%} of episodes)")
        lines.append("")
    else:
        lines.append(f"*{milestones.get('note', 'Not available')}*")
        lines.append("")

    # Section 5: Episode health
    lines.append("## S5: Episode Health\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Episodes | {episode.get('n_episodes', 0)} |")
    if "length_mean" in episode:
        lines.append(f"| Length mean | {episode['length_mean']:.0f} steps |")
        lines.append(f"| Length range | {episode.get('length_min', 0)}-{episode.get('length_max', 0)} |")
    if "envs_seen" in episode:
        lines.append(f"| Envs used | {episode['envs_seen']} |")
    if "min_range_best" in episode:
        lines.append(f"| Min range (best) | {episode['min_range_best']:.1f} m |")
        lines.append(f"| Min range (mean) | {episode['min_range_mean']:.1f} m |")
    lines.append("")

    # Section 6: Learned vs Baseline (v12, C006 P4 renamed)
    lines.append("## S6: Learned vs Baseline\n")
    if learned_baseline and learned_baseline.get("available"):
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Tier 4 mean residual | {learned_baseline.get('tier4_mean_residual_overall', 0):.4f} |")
        lines.append(f"| Learned tier 4 fraction | {learned_baseline.get('learned_tier4_fraction_overall', 0):.1%} |")
        corr = learned_baseline.get("residual_prox_correlation_mean", float("nan"))
        corr_str = f"{corr:.4f}" if corr == corr else "N/A"
        lines.append(f"| Residual-prox correlation | {corr_str} |")
        if "tier3_mean_residual_overall" in learned_baseline:
            lines.append(f"| Tier 3 mean residual | {learned_baseline['tier3_mean_residual_overall']:.4f} |")
        pt_rate = learned_baseline.get("baseline_passthrough_rate", 0)
        pt_eps = learned_baseline.get("baseline_passthrough_episodes", 0)
        lines.append(f"| Baseline passthrough | {pt_eps} episodes ({pt_rate:.0%}) |")
        if pt_rate > 0.5:
            lines.append(f"\n**WARNING**: >50% of episodes show BASELINE PASSTHROUGH "
                         f"(learned_precontact_tier4_fraction < 10%). The policy is not "
                         f"deviating from zero-residual.")
    else:
        note = learned_baseline.get("note", "Not available") if learned_baseline else "Not available"
        lines.append(f"*{note}*")
    lines.append("")

    # Section 7: Actor Loss Acceleration (C006 P9)
    lines.append("## S7: Actor Loss Acceleration\n")
    if actor_loss_accel and actor_loss_accel.get("available"):
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Actor loss (final) | {actor_loss_accel.get('actor_loss_final', 0):.4f} |")
        lines.append(f"| Max consecutive +d²/de² | {actor_loss_accel.get('max_consecutive_pos_d2', 0)} |")
        if "critic_loss_final" in actor_loss_accel:
            lines.append(f"| Critic loss (final) | {actor_loss_accel['critic_loss_final']:.4f} |")
            lines.append(f"| Critic loss (mean) | {actor_loss_accel['critic_loss_mean']:.4f} |")
        if actor_loss_accel.get("actor_loss_diverging"):
            lines.append(f"\n**WARNING**: Actor loss 2nd derivative positive for "
                         f"{actor_loss_accel['max_consecutive_pos_d2']} consecutive episodes. "
                         f"Potential Q-value explosion.")
    else:
        note = actor_loss_accel.get("note", "Not available") if actor_loss_accel else "Not available"
        lines.append(f"*{note}*")
    lines.append("")

    # Decision section
    lines.append("---\n")
    lines.append(f"## Decision: **{decision}**\n")
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="DRL Post-Campaign Analysis",
    )
    parser.add_argument(
        "--log-dir", type=str, required=True,
        help="Directory containing metrics.csv and diagnostics CSV",
    )
    parser.add_argument(
        "--algorithm", type=str, default="td3",
        choices=["td3", "sac", "ppo"],
        help="Algorithm used in the campaign (default: td3). "
             "Determines which diagnostics CSV to read.",
    )
    parser.add_argument(
        "--force-zero-residual", action="store_true", default=False,
        help="Analyze as FZR baseline measurement run (suppresses learning-signal analysis).",
    )
    args = parser.parse_args()
    log_dir = Path(args.log_dir)
    algorithm = args.algorithm

    if not log_dir.exists():
        print(f"ERROR: Log directory not found: {log_dir}", file=sys.stderr)
        return 1

    # Load data
    metrics_df = load_metrics(log_dir)

    if metrics_df is None:
        print(f"ERROR: No metrics.csv in {log_dir}", file=sys.stderr)
        return 1

    # --- FZR detection: explicit flag OR config_snapshot.json ---
    fzr_mode = args.force_zero_residual
    if not fzr_mode:
        snap_path = log_dir / "config_snapshot.json"
        if snap_path.exists():
            with open(snap_path) as _f:
                _snap = json.load(_f)
            _exp = _snap.get("expected", _snap)
            fzr_mode = bool(_exp.get("force_zero_residual", False))

    if fzr_mode:
        n = len(metrics_df)
        col = "actual_M3b_achieved"
        k_full = int(coerce_achievement_series(metrics_df[col]).sum()) if col in metrics_df.columns else 0
        p_full, ci_lo, ci_hi = _wilson_ci(k_full, n)
        lines = [
            "# FZR Baseline Measurement Report\n",
            f"**Episodes**: {n}\n",
            "## M3b Counts\n",
            f"- Full sample: {k_full}/{n} = {p_full*100:.4f}%  "
            f"Wilson 95% CI [{ci_lo*100:.4f}%, {ci_hi*100:.4f}%]",
        ]
        if n >= 96:
            k_a = int(coerce_achievement_series(metrics_df[col].iloc[:96]).sum()) if col in metrics_df.columns else 0
            p_a, lo_a, hi_a = _wilson_ci(k_a, 96)
            lines.append(
                f"- Slice A (first 96): {k_a}/96 = {p_a*100:.4f}%  "
                f"Wilson 95% CI [{lo_a*100:.4f}%, {hi_a*100:.4f}%]"
            )
        dbb_col = "actual_M3b_dwell_s"
        if dbb_col in metrics_df.columns:
            lines.append(f"- DBB mean: {metrics_df[dbb_col].mean():.3f} s")
        lines.append(
            "\n## Suppressed\nFZR mode detected; learning-signal metrics suppressed "
            "(r_prox, policy_speed_delta, tier_progression are meaningless under "
            "learning_rate=0.0 + learning_starts > total_timesteps).\n"
            "\n**DECISION: BASELINE_MEASUREMENT**"
        )
        report = "\n".join(lines)
        (log_dir / "campaign_analysis.md").write_text(report)
        print(f"FZR baseline: {k_full}/{n} M3b ({p_full*100:.4f}%), "
              f"CI [{ci_lo*100:.4f}%, {ci_hi*100:.4f}%]")
        print(f"Report written to: {log_dir / 'campaign_analysis.md'}")
        print(f"\n{'='*50}")
        print(f"  DECISION: BASELINE_MEASUREMENT")
        print(f"{'='*50}")
        return 0

    print(f"Analyzing {len(metrics_df)} episodes from {log_dir} (algorithm={algorithm})...")

    # Load algorithm-specific diagnostics
    if algorithm in ("td3", "sac"):
        offpolicy_diag_df = load_offpolicy_diagnostics(log_dir)
        if offpolicy_diag_df is not None:
            print(f"  Off-policy diagnostics: {len(offpolicy_diag_df)} episodes")
        policy = analyze_offpolicy_stability(offpolicy_diag_df)
    else:
        diag_df = load_policy_diagnostics(log_dir)
        if diag_df is not None:
            print(f"  Policy diagnostics: {len(diag_df)} rollouts")
        policy = analyze_policy_stability(diag_df)

    # Run 6 analysis sections + actor loss acceleration check (C006 P9)
    sigstop = analyze_sigstop(metrics_df)
    reward = analyze_reward_trend(metrics_df)
    milestones_result = analyze_milestones(metrics_df)
    episode = analyze_episode_health(metrics_df)
    learned_baseline = analyze_learned_vs_baseline(metrics_df)
    actor_loss_accel = check_actor_loss_acceleration(str(log_dir))

    # Compute decision
    decision, reasons = compute_decision(reward, policy, milestones_result, episode)

    # C006 P9: Escalate if actor loss is diverging
    if actor_loss_accel.get("actor_loss_diverging"):
        reasons.append(
            f"Actor loss 2nd derivative positive for "
            f"{actor_loss_accel['max_consecutive_pos_d2']} consecutive episodes "
            f"(Q-value explosion risk)"
        )

    # Generate report
    report = generate_report(
        log_dir, sigstop, reward, policy, milestones_result, episode,
        decision, reasons, learned_baseline=learned_baseline,
        actor_loss_accel=actor_loss_accel,
    )

    # Write markdown report
    output_path = log_dir / "campaign_analysis.md"
    output_path.write_text(report)
    print(f"\nReport written to: {output_path}")

    # Print decision to stdout
    print(f"\n{'='*50}")
    print(f"  DECISION: {decision}")
    for r in reasons:
        print(f"    - {r}")
    print(f"{'='*50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
