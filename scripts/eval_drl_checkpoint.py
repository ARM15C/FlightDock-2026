#!/usr/bin/env python3
"""DRL checkpoint evaluation — deterministic inference with obs normalization.

Loads a TD3 checkpoint + VecNormalize stats, runs N episodes with
deterministic=True (no exploration noise) via raw Gymnasium API.
Same CSV columns, metrics extraction, and summary format as
eval_baseline_zero_residual.py.

Usage:
    python scripts/eval_drl_checkpoint.py \
        --checkpoint ~/FlightDock/checkpoints/path1_campaign1/model_306000_steps.zip \
        --stage 0 --n-episodes 10 --speedup 2.0 --env-id 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import pickle
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Ensure src/ and tests/ are on path (same pattern as eval_baseline_zero_residual.py)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("src", "tests"):
    p = os.path.join(_REPO_ROOT, d)
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Graceful shutdown flag
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    logger.warning("Signal %d received — finishing current episode then exiting", signum)
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# CSV column order — identical to eval_baseline_zero_residual.py
CSV_COLUMNS = [
    "episode", "steps", "total_reward", "terminal_type", "fsm_state",
    "actual_M2_achieved", "actual_M2_dwell_s",
    "actual_M3a_coarse_achieved", "actual_M3a_coarse_dwell_s",
    "actual_M3a_fine_achieved", "actual_M3a_fine_dwell_s",
    "actual_M3b_achieved", "actual_M3b_dwell_s",
    "min_range_m", "actual_min_boom_tip_range_m",
    "dwell_M2_s", "dwell_M3a_coarse_s", "dwell_M3a_fine_s", "dwell_M3b_s",
    "time_to_M2_s", "time_to_M3a_coarse_s", "time_to_M3a_fine_s", "time_to_M3b_s",
    "max_behind_overshoot_m", "max_cross_excursion_m",
    # TERMDEMOTE-767. Appended at the END of the list so every pre-existing
    # column keeps its position -- a reader that indexes by position, not by
    # name, must not silently shift. Numeric only; ticks > 0 is the predicate.
    "demoted_collision_ticks", "demoted_collision_intervals",
    "demoted_collision_first_s", "demoted_collision_last_s",
    "demoted_airspeed_ticks", "demoted_airspeed_intervals",
    "demoted_airspeed_first_s", "demoted_airspeed_last_s",
    "leader_profile",
    # T-393: every leg must carry its own
    # independent-variable settings. APPENDED, never reordered, so existing
    # readers keep working. wind_turb is deliberately NOT among them: it is
    # the legacy knob, inert on the json backend, and recording it as the
    # turbulence is exactly the misreport this row exists to prevent.
    "wind_speed_ms", "wind_from_deg", "turbulence_sigma_ms",
    "gps_cep_m", "v3_cell_label",
    # TELEMFIX-771. APPENDED at the very END, same rule as the blocks above.
    # These were emitted into `info` by the env from the day the freshness
    # guard landed and reached disk on NO leg, because the row builder below
    # copied only the "demoted_" prefix. The 4-leg smoke of 2026-08-20 had to
    # recover the equivalent offline from MAVLink wire stamps. A counter that
    # nobody can read is not evidence. Detector: test_abstain_columns.py.
    "telem_abstain_airspeed", "telem_abstain_alt", "telem_abstain_descent",
]

PER_STEP_COLUMNS = [
    "episode_idx", "step_idx", "behind_m", "cross_m", "below_m",
    "in_m3b", "m3b_dwell_s", "range_m",
]


def _extract_episode_metrics(ep_num, step_count, total_reward, info):
    """Extract flat dict from terminal info for CSV row."""
    return {
        "episode": ep_num,
        "steps": step_count,
        "total_reward": round(total_reward, 4),
        "terminal_type": info.get("terminal_type", "unknown"),
        # TERMDEMOTE-767: no .get() default here on purpose. If the env did
        # not emit these, the leg was flown by a pre-TERMDEMOTE env and the
        # KeyError says so loudly. A default of 0 would forge "no event".
        **{k: info[k] for k in info
           if k.startswith("demoted_") or k.startswith("telem_abstain_")},
        "fsm_state": info.get("fsm_state", "UNKNOWN"),
        "actual_M2_achieved": info.get("actual_M2_achieved", False),
        "actual_M2_dwell_s": round(info.get("actual_M2_dwell_s", 0.0), 3),
        "actual_M3a_coarse_achieved": info.get("actual_M3a_coarse_achieved", False),
        "actual_M3a_coarse_dwell_s": round(info.get("actual_M3a_coarse_dwell_s", 0.0), 3),
        "actual_M3a_fine_achieved": info.get("actual_M3a_fine_achieved", False),
        "actual_M3a_fine_dwell_s": round(info.get("actual_M3a_fine_dwell_s", 0.0), 3),
        "actual_M3b_achieved": info.get("actual_M3b_achieved", False),
        "actual_M3b_dwell_s": round(info.get("actual_M3b_dwell_s", 0.0), 3),
        "min_range_m": round(info.get("min_range_m", 0.0), 3),
        "actual_min_boom_tip_range_m": round(info.get("actual_min_boom_tip_range_m", -1.0), 3),
        "dwell_M2_s": round(info.get("dwell_M2_s", 0.0), 3),
        "dwell_M3a_coarse_s": round(info.get("dwell_M3a_coarse_s", 0.0), 3),
        "dwell_M3a_fine_s": round(info.get("dwell_M3a_fine_s", 0.0), 3),
        "dwell_M3b_s": round(info.get("dwell_M3b_s", 0.0), 3),
        "time_to_M2_s": round(info.get("time_to_M2_s", -1.0), 3),
        "time_to_M3a_coarse_s": round(info.get("time_to_M3a_coarse_s", -1.0), 3),
        "time_to_M3a_fine_s": round(info.get("time_to_M3a_fine_s", -1.0), 3),
        "time_to_M3b_s": round(info.get("time_to_M3b_s", -1.0), 3),
        "max_behind_overshoot_m": round(info.get("max_behind_overshoot_m", 0.0), 3),
        "max_cross_excursion_m": round(info.get("max_cross_excursion_m", 0.0), 3),
        "leader_profile": info.get("leader_profile", "racetrack"),
        # T-393 independent-variable witness. UNSAMPLED is not CLEAN: a missing
        # value writes the empty string, never a plausible-looking default, so
        # a leg that failed to report its conditions can never be mistaken for
        # one that flew in calm air.
        "wind_speed_ms": info.get("wind_speed_ms", ""),
        "wind_from_deg": info.get("wind_dir_deg", ""),
        "turbulence_sigma_ms": ";".join(
            f"{float(s):.3f}" for s in (info.get("wind_sigma_ms") or [])),
        "gps_cep_m": info.get("gps_accuracy", ""),
        "v3_cell_label": info.get("v3_cell_label", ""),
    }


def _print_summary(episodes_data, stage, checkpoint_path):
    """Compute and print aggregate statistics."""
    n = len(episodes_data)
    if n == 0:
        logger.warning("No episodes completed — no summary to print.")
        return ""

    lines = []
    lines.append("=" * 70)
    lines.append("DRL CHECKPOINT EVALUATION")
    lines.append(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    lines.append(f"Stage: {stage}  |  Episodes completed: {n}")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # 1. Terminal type distribution
    term_types = [e["terminal_type"] for e in episodes_data]
    unique = sorted(set(term_types))
    lines.append("\n--- Terminal Type Distribution ---")
    for t in unique:
        count = term_types.count(t)
        lines.append(f"  {t}: {count}/{n} ({100*count/n:.1f}%)")

    # 2. Milestone achievement rates
    lines.append("\n--- Milestone Achievement Rates ---")
    for milestone in ["M2", "M3a_coarse", "M3a_fine", "M3b"]:
        key = f"actual_{milestone}_achieved"
        achieved = sum(1 for e in episodes_data if e[key])
        lines.append(f"  {milestone}: {achieved}/{n} ({100*achieved/n:.1f}%)")

    # 3. M3b dwell statistics
    m3b_dwells_all = [e["actual_M3b_dwell_s"] for e in episodes_data]
    m3b_dwells_achievers = [d for d in m3b_dwells_all if d >= 3.0]
    lines.append("\n--- M3b Dwell Statistics ---")
    lines.append(f"  All episodes:  mean={np.mean(m3b_dwells_all):.3f}s, "
                 f"std={np.std(m3b_dwells_all):.3f}s, "
                 f"min={np.min(m3b_dwells_all):.3f}s, "
                 f"max={np.max(m3b_dwells_all):.3f}s")
    if m3b_dwells_achievers:
        lines.append(f"  Achievers only: mean={np.mean(m3b_dwells_achievers):.3f}s, "
                     f"std={np.std(m3b_dwells_achievers):.3f}s, "
                     f"min={np.min(m3b_dwells_achievers):.3f}s, "
                     f"max={np.max(m3b_dwells_achievers):.3f}s")
    else:
        lines.append("  Achievers only: N/A (no episodes achieved M3b)")

    # 4. M3b corridor entry rate (episodes with any dwell > 0)
    corridor_entries = sum(1 for d in m3b_dwells_all if d > 0)
    lines.append(f"\n--- M3b Corridor Entry ---")
    lines.append(f"  Episodes with dwell > 0: {corridor_entries}/{n} ({100*corridor_entries/n:.1f}%)")

    # 5. Proximity statistics
    boom_ranges = [e["actual_min_boom_tip_range_m"] for e in episodes_data
                   if e["actual_min_boom_tip_range_m"] >= 0]
    lines.append("\n--- Proximity (Boom-Tip Min Range) ---")
    if boom_ranges:
        lines.append(f"  mean={np.mean(boom_ranges):.3f}m, "
                     f"min={np.min(boom_ranges):.3f}m, "
                     f"max={np.max(boom_ranges):.3f}m")
    else:
        lines.append("  N/A (no valid boom-tip range data)")

    # 6. Formation loss
    formation_loss = sum(1 for e in episodes_data if e["min_range_m"] > 10.0)
    lines.append(f"\n--- Formation Loss ---")
    lines.append(f"  Episodes with min_range > 10m: {formation_loss}/{n} ({100*formation_loss/n:.1f}%)")

    # 7. Total reward statistics
    rewards = [e["total_reward"] for e in episodes_data]
    lines.append(f"\n--- Total Reward ---")
    lines.append(f"  mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}, "
                 f"min={np.min(rewards):.3f}, max={np.max(rewards):.3f}")

    # 8. Comparison to Stage 0 baseline
    m3b_rate = sum(1 for e in episodes_data if e["actual_M3b_achieved"]) / n
    lines.append(f"\n--- Stage 0 Baseline Comparison ---")
    lines.append(
        "  Stage 0 reference: [DEPRECATED pre-UWB-restoration literal; "
        "pending fresh collection under restored substrate]"
    )
    if boom_ranges:
        lines.append(f"  Stage {stage} result: {100*m3b_rate:.1f}% M3b, "
                     f"{np.mean(m3b_dwells_all):.3f}s mean dwell, "
                     f"{np.mean(boom_ranges):.3f}m mean proximity")
    else:
        lines.append(f"  Stage {stage} result: {100*m3b_rate:.1f}% M3b, "
                     f"{np.mean(m3b_dwells_all):.3f}s mean dwell, N/A proximity")

    # 9. Verdict
    lines.append(f"\n--- Verdict ---")
    m3b_pct = 100 * m3b_rate
    if m3b_pct >= 30:
        verdict = "VIABLE"
        detail = f"M3b rate {m3b_pct:.1f}% >= 30% threshold"
    elif m3b_pct >= 10:
        verdict = "MARGINAL"
        detail = f"M3b rate {m3b_pct:.1f}% in 10-30% range"
    else:
        verdict = "SUSPECT"
        detail = f"M3b rate {m3b_pct:.1f}% < 10% threshold"
    lines.append(f"  {verdict}: {detail}")

    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    print(summary_text)
    return summary_text, verdict, m3b_pct


def _load_obs_normalizer(vecnormalize_path):
    """Load VecNormalize stats from pickle file.

    Returns (obs_rms_mean, obs_rms_var, clip_obs, epsilon) tuple.
    Handles both full VecNormalize object and dict formats.
    """
    with open(vecnormalize_path, "rb") as f:
        vn = pickle.load(f)

    if isinstance(vn, dict):
        obs_rms = vn["obs_rms"]
        clip_obs = vn.get("clip_obs", 10.0)
        epsilon = vn.get("epsilon", 1e-8)
    else:
        obs_rms = vn.obs_rms
        clip_obs = vn.clip_obs
        epsilon = vn.epsilon

    logger.info("VecNormalize loaded: obs_dim=%d, clip_obs=%.1f, epsilon=%.0e",
                obs_rms.mean.shape[0], clip_obs, epsilon)
    return obs_rms.mean, obs_rms.var, clip_obs, epsilon


def main():
    parser = argparse.ArgumentParser(
        description="DRL checkpoint evaluation — deterministic inference with obs normalization",
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model ZIP file")
    parser.add_argument("--vecnormalize", type=str, default=None,
                        help="Path to VecNormalize PKL (auto-derived from --checkpoint if omitted)")
    parser.add_argument("--stage", type=int, default=0,
                        help="Curriculum stage (determines wind/turb/GPS bounds)")
    parser.add_argument("--n-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--speedup", type=float, default=2.0,
                        help="SITL speedup factor")
    parser.add_argument("--env-id", type=int, default=0,
                        help="Port isolation (env_id offset)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible domain randomization")
    parser.add_argument("--warmup-sim-s", type=float, default=1200.0,
                        help="Max sim-time for warmup phase (default: 1200s)")
    parser.add_argument("--log-per-step", action="store_true", default=False,
                        help="Write per-step pose CSV for H1 dwell-phase analysis")
    parser.add_argument("--force-zero-residual", action="store_true", default=False,
                        help="T-57: inject a zero residual through the normal path "
                             "(config.INJECT_ZERO_ACTION) so the eval assembly reduces "
                             "to the baseline FSM — the Amendment-4 Part A "
                             "assembly-equivalence gate. Mirrors train.py --inject-zero-"
                             "action. Default off => policy acts as normal.")
    args = parser.parse_args()

    # FIXAWIRE-719: assert the EVAL contract HERE — at the point of flight,
    # before `drl.env_formation_sitl` is imported below. Placement is
    # load-bearing: _FD_RAW_ALIGN_CENTER is read at MODULE IMPORT, so a check
    # after the import would judge an environment whose value had already been
    # baked in. Unconditional by (the relay's
    # FD_CURRICULUM_V3 guard let a bare-shell launch skip the contract; this
    # path is the ARM-F/policy harness and had no check at all).
    from drl import env_contract          # 'drl', not 'src.drl': this file
    env_contract.assert_contract(env_contract.EVAL)   # puts src/ on sys.path

    # Derive vecnormalize path from checkpoint if not provided
    if args.vecnormalize is None:
        vn_base = os.path.basename(args.checkpoint).replace(
            "model_", "vecnormalize_").replace(".zip", ".pkl")
        args.vecnormalize = os.path.join(os.path.dirname(args.checkpoint), vn_base)

    # Verify files exist
    for label, path in [("checkpoint", args.checkpoint), ("vecnormalize", args.vecnormalize)]:
        if not os.path.isfile(path):
            logger.error("%s not found: %s", label, path)
            sys.exit(1)

    if args.output_dir is None:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.output_dir = os.path.join(_REPO_ROOT, f"logs/checkpoint_eval_{ckpt_name}")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("VecNormalize: %s", args.vecnormalize)

    # Deferred imports (after sys.path setup)
    from stable_baselines3 import TD3
    from stable_baselines3.common.utils import set_random_seed

    from drl import config

    # T-57: Amendment-4 Part A. Zero residual through the injection site
    # (env_formation_sitl.py:1614) reduces the eval assembly to the baseline
    # FSM. Set on config BEFORE the env is constructed, exactly as
    # train.py:1664 does. Default off => this branch never runs => non-forced
    # evals are behaviour-identical.
    if args.force_zero_residual:
        config.INJECT_ZERO_ACTION = True
        logger.info("[T-57 OVERRIDE] INJECT_ZERO_ACTION -> True "
                    "(Part A assembly-equivalence: eval assembly == baseline FSM)")
    from drl.curriculum import CurriculumManager
    from drl.env_formation_sitl import FormationSITLEnv

    # Phase G seed chain: SB3 set_random_seed helper seeds random+numpy+torch
    # uniformly. Preferred over manual blocks (operator M4 consistency with
    # train.py:1115-1116).
    if args.seed is not None:
        set_random_seed(args.seed, using_cuda=torch.cuda.is_available())
        logger.info("Random seed: %d (via SB3 set_random_seed, cuda=%s)",
                    args.seed, torch.cuda.is_available())

    # --- Phase 3b child-side RNG dump for smoke test V1 ---
    # Captures RNG state immediately after set_random_seed(), before any
    # SITL I/O or env construction. Proves child-process state matches
    # expectations for a given seed input.
    if args.seed is not None:
        rng_state = {
            "python_random": str(random.getstate()),
            "numpy_random": str(np.random.get_state()),
            "torch_seed_initial_state": str(torch.initial_seed()),
        }
        state_hash = hashlib.sha256(
            json.dumps(rng_state, sort_keys=True).encode()
        ).hexdigest()
        rng_dump_path = Path(args.output_dir) / f"rng_state_child_{args.env_id}.json"
        rng_dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rng_dump_path, "w") as f:
            json.dump({
                "seed_value": args.seed,
                "env_id": args.env_id,
                "rng_state_hash": state_hash,
                "process_pid": os.getpid(),
                "timestamp_unix": int(time.time()),
            }, f, indent=2)
        logger.info("RNG state dumped: %s (hash=%s)",
                    rng_dump_path, state_hash[:12])
    # --- end Phase 3b RNG dump ---

    # Load model
    model = TD3.load(args.checkpoint)
    logger.info("TD3 model loaded: obs_space=%s, action_space=%s",
                model.observation_space.shape, model.action_space.shape)

    # Load obs normalization stats
    obs_mean, obs_var, clip_obs, epsilon = _load_obs_normalizer(args.vecnormalize)

    # Scale wall-clock warmup timeout to match sim-time limit
    config.WARMUP_TIMEOUT_S = max(
        config.WARMUP_TIMEOUT_S,
        args.warmup_sim_s / args.speedup * 1.2,
    )
    config.MAX_WARMUP_RETRIES = 5
    logger.info("Warmup limits: sim=%.0fs, wall=%.0fs, retries=%d",
                args.warmup_sim_s, config.WARMUP_TIMEOUT_S, config.MAX_WARMUP_RETRIES)

    # Create curriculum manager locked to specified stage
    curriculum = CurriculumManager(initial_stage=args.stage)
    # T-393: report the REAL turbulence. The legacy `wind_turb` knob is INERT
    # on the json backend (curriculum.py:412-414) and reads 0.0 on cells that
    # are genuinely turbulent -- printing it as "turb=" told operators these
    # legs flew in still air. The flown turbulence is the FDM Dryden sigma.
    _st = config.CURRICULUM_STAGES[args.stage]
    _sig = getattr(_st, "wind_sigma_ms", ()) or ()
    logger.info(
        "Curriculum stage: %d (%s) wind_max=%.1f m/s from %.0f deg, "
        "turbulence_sigma_ms=%s, gps_cep_m=%.2f  [legacy SIM_WIND_TURB knob="
        "%.1f, INERT on json backend -- not the turbulence]",
        args.stage, config.STAGE_LABELS_V3.get(args.stage, "?"),
        _st.wind_max_ms, getattr(_st, "wind_from_deg", 0.0),
        ",".join(f"{s:.3f}" for s in _sig) if _sig else "NONE",
        _st.gps_accuracy, _st.wind_turb)

    # Create environment
    env = FormationSITLEnv(
        env_id=args.env_id,
        speedup=args.speedup,
        curriculum_manager=curriculum,
        ui_mode="none",
        warmup_max_sim_s=args.warmup_sim_s,
        # T-51: the V3 per-episode RNG derives from the CONSTRUCTOR seed
        # (env:1131 gates on _initial_seed, set only at :646); reset(seed=)
        # feeds Gymnasium np_random and never reaches it. Same convention as
        # train.py:744-751. None stays None — non-seeded evals unchanged.
        seed=args.seed,
        # T-55: without out_dir the controller's _out_dir stays None and it
        # writes NONE of TRANSITION_SNAPSHOT.log / speed_arbitration.csv /
        # run_summary.json / success_metrics.json (ctrl:3477 fail-silent
        # guard) — the leg flies but no sealed-gate scorer can read it.
        # reset() nests per-episode at <out_dir>/episode_NNN_envM/ (env:634),
        # the layout the certified corpus already uses.
        out_dir=args.output_dir,
    )

    # Set safety range for stage (before first reset)
    env.set_safety_range(args.stage)
    logger.info("Safety range: %.0f m (stage %d)", env._safety_range_m, args.stage)

    episodes_data = []
    # Phase 3b: env-id-ify CSV filename to prevent collision when 4 parallel
    # processes share a single --output-dir.
    csv_path = os.path.join(args.output_dir, f"metrics_{args.env_id}.csv")

    # Phase 3b: capture raw info dicts for manifest
    _raw_first_info: dict = {}
    _raw_last_info: dict = {}

    warmup_failures = 0
    max_warmup_failures = 10

    try:
        # Per-step logging (R1: same dir as episode CSV, R2: checkpoint-based name, R3: inside try)
        per_step_fh = None
        per_step_writer = None
        per_step_path = None
        _per_step_validated = False
        if args.log_per_step:
            ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
            per_step_path = os.path.join(args.output_dir, f"{ckpt_stem}_per_step.csv")
            per_step_fh = open(per_step_path, "w", newline="")
            per_step_writer = csv.DictWriter(per_step_fh, fieldnames=PER_STEP_COLUMNS)
            per_step_writer.writeheader()
            logger.info("Per-step logging enabled: %s", per_step_path)

        ep_num = 0
        consecutive_failures = 0
        first_reset = True
        while ep_num < args.n_episodes:
            if _shutdown:
                logger.info("Shutdown requested — stopping after %d episodes", ep_num)
                break

            try:
                # Phase G: seed env's np_random on first reset only per
                # Gymnasium convention. Subsequent resets use the seeded stream.
                seed_arg = args.seed if first_reset else None
                first_reset = False
                obs, info = env.reset(seed=seed_arg)
            except RuntimeError as e:
                warmup_failures += 1
                consecutive_failures += 1
                logger.warning(
                    "Warmup failure %d (consecutive=%d): %s — retrying",
                    warmup_failures, consecutive_failures, e,
                )
                if consecutive_failures >= max_warmup_failures:
                    logger.error(
                        "Aborting: %d consecutive warmup failures", consecutive_failures
                    )
                    break
                continue

            consecutive_failures = 0
            done = False
            step_count = 0
            total_reward = 0.0

            while not done:
                # Apply VecNormalize obs normalization
                obs_norm = np.clip(
                    (obs - obs_mean) / np.sqrt(obs_var + epsilon),
                    -clip_obs, clip_obs,
                ).astype(np.float32)

                # Deterministic inference (no exploration noise)
                action, _ = model.predict(obs_norm, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1
                total_reward += float(reward)

                if per_step_writer is not None:
                    # R5: fail-loud on first step if env contract changed
                    if not _per_step_validated:
                        _required = ["behind_m", "cross_m", "below_m",
                                     "in_m3b", "m3b_dwell_s", "range_m"]
                        _missing = [k for k in _required if k not in info]
                        if _missing:
                            raise RuntimeError(
                                f"Per-step logging requires env info keys: "
                                f"missing {_missing}. Env contract changed; "
                                f"update PER_STEP_COLUMNS or env_formation_sitl."
                            )
                        _per_step_validated = True
                    per_step_writer.writerow({
                        "episode_idx": ep_num + 1,
                        "step_idx": step_count,
                        "behind_m": round(info.get("behind_m", 0.0), 6),
                        "cross_m": round(info.get("cross_m", 0.0), 6),
                        "below_m": round(info.get("below_m", 0.0), 6),
                        "in_m3b": int(bool(info.get("in_m3b", False))),
                        "m3b_dwell_s": round(info.get("m3b_dwell_s", 0.0), 4),
                        "range_m": round(info.get("range_m", 0.0), 6),
                    })

            ep_num += 1

            # Extract metrics from terminal info
            row = _extract_episode_metrics(ep_num, step_count, total_reward, info)
            episodes_data.append(row)

            # Phase 3b: retain raw info dict for manifest
            if not _raw_first_info:
                _raw_first_info = dict(info)
            _raw_last_info = dict(info)

            m3b_tag = "M3b" if row["actual_M3b_achieved"] else "---"
            logger.info(
                "Episode %d/%d: %d steps, reward=%.3f, %s, "
                "M3b_dwell=%.2fs, boom_prox=%.3fm (warmup_fails=%d)",
                ep_num, args.n_episodes, step_count, total_reward,
                m3b_tag, row["actual_M3b_dwell_s"],
                row["actual_min_boom_tip_range_m"],
                warmup_failures,
            )

            # Write CSV incrementally (rewrite each episode for crash safety)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(episodes_data)

            if per_step_fh is not None:
                per_step_fh.flush()

    finally:
        env.close()
        if per_step_fh is not None:
            per_step_fh.close()
            logger.info("Per-step CSV: %s", per_step_path)
        if warmup_failures > 0:
            logger.info("Total warmup failures across run: %d", warmup_failures)

        # --- Phase 3b per-process manifest ---
        # Written in finally so it persists even on exceptions.
        # Spec's nested reward_components is traversed for reward-key detection.
        _rc = {}
        if _raw_last_info:
            _rc_obj = _raw_last_info.get("reward_components", {})
            if isinstance(_rc_obj, dict):
                _rc = _rc_obj
        top_level_keys = list(_raw_last_info.keys()) if _raw_last_info else []
        nested_rc_keys = list(_rc.keys())
        reward_keys_present = sorted(set(
            k for k in (top_level_keys + nested_rc_keys)
            if k.startswith("r_") or k in [
                "reward", "total_mean",
                "ratio_milestone_to_ref_mean",
                "ratio_smoothness_abs_to_total_mean",
                "leader_profile",
            ]
        ))

        vecnormalize_path = args.vecnormalize or None
        vecnormalize_sha256 = (
            hashlib.sha256(Path(vecnormalize_path).read_bytes()).hexdigest()
            if vecnormalize_path and Path(vecnormalize_path).exists()
            else None
        )

        # eval_drl_checkpoint.py loads VN stats manually (no VecEnv wrapper);
        # vec_env.training is not applicable. Record mode for transparency.
        stage_cfg = config.CURRICULUM_STAGES[args.stage]
        manifest_path = Path(args.output_dir) / f"manifest_{args.env_id}.json"
        manifest = {
            "seed_value": args.seed,
            "env_id": args.env_id,
            "stage": args.stage,
            "n_episodes_completed": len(episodes_data),
            "checkpoint_path": str(args.checkpoint),
            "vecnormalize_path": str(vecnormalize_path) if vecnormalize_path else None,
            "vecnormalize_sha256": vecnormalize_sha256,
            "vecnormalize_training_flag": None,
            "vecnormalize_mode": "manual_tuple",
            "reward_keys_present_in_info": reward_keys_present,
            "first_episode_dr_values": {
                "wind_speed_ms": (_raw_first_info.get("wind_speed_ms")
                                  if _raw_first_info else None),
                "wind_dir_deg": (_raw_first_info.get("wind_dir_deg")
                                 if _raw_first_info else None),
                "wind_turb_LEGACY_INERT_ON_JSON": (
                    _raw_first_info.get("wind_turb")
                    if _raw_first_info else None),
                # T-393: the REAL flown turbulence (FDM Dryden sigma triple).
                "turbulence_sigma_ms": (
                    _raw_first_info.get("wind_sigma_ms")
                    if _raw_first_info else None),
                "gps_accuracy": (_raw_first_info.get("gps_accuracy")
                                 if _raw_first_info else None),
                "leader_profile": (_raw_first_info.get("leader_profile")
                                   if _raw_first_info else None),
            },
            "stage_config_dr_bounds": {
                "wind_max_ms": stage_cfg.wind_max_ms,
                "wind_from_deg": getattr(stage_cfg, "wind_from_deg", None),
                # T-393: name the legacy knob for what it is, and record the
                # sigma that actually reached the FDM alongside it. Before
                # this, a leg's only turbulence record read 0.0 on a cell
                # flying sigma 0.20 -- an unsampled condition that LOOKED
                # like calm air.
                "wind_turb_LEGACY_INERT_ON_JSON": stage_cfg.wind_turb,
                "turbulence_sigma_ms": list(
                    getattr(stage_cfg, "wind_sigma_ms", ()) or ()),
                "gps_accuracy": stage_cfg.gps_accuracy,
            },
            # T434AMEND hardening (c), 2026-08-16: record the LIVE venue flags
            # so a banked leg self-witnesses which judge/guard/plant it flew.
            # Read from os.environ, never from config — config resolves at
            # import and a manifest that echoed the default would witness
            # nothing. Absent => the flag rode its source default.
            "venue_env": {k: os.environ.get(k) for k in (
                "FD_COLLISION_TRUTH_CHANNEL",   # SAFETY_COLLISION judge channel
                "FD_GUARD_CEILING_CHANNEL",     # ceiling-guard input channel
                "FD_CEILING_SHAPE_ENABLE",      # SHAPE-585 endpoint shaping
                "FD_FSM_NESTING_REPAIR",        # FSMDWELL-503 plant repair
                "ARMB_WAVE",                    # ARMBTOP-F wave provenance
                # TERMDEMOTE-767 / R8. Without these a re-score cannot tell
                # a demoted leg from a pre-demotion one and every scorer
                # silently mixes two populations.
                "FD_DEMOTE_TERMINALS",          # which terminals were demoted
                "FD_FORMATION_LOST_TERM",       # R8: must be absent on EVAL
                # NOPASSTERM-770: which terminals were live and at what line.
                "FD_SAFETY_TERMINALS",
                "FD_NOPASS_TERM_AHEAD_M",
                "FD_NOPASS_TERM_PERSIST_S",
                "FD_NOPASS_TERM_OFFAXIS_M",
            )},
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info("Manifest: %s (%d reward_keys)",
                    manifest_path, len(reward_keys_present))
        # --- end Phase 3b manifest ---

    # Print and save summary
    logger.info("Writing summary to %s", args.output_dir)
    result = _print_summary(episodes_data, args.stage, args.checkpoint)
    if result:
        summary_text, verdict, m3b_pct = result
        summary_path = os.path.join(args.output_dir, f"summary_eid{args.env_id}.txt")
        with open(summary_path, "w") as f:
            f.write(summary_text + "\n")

        # Write JSON results for downstream consumption
        n = len(episodes_data)
        boom_ranges = [e["actual_min_boom_tip_range_m"] for e in episodes_data
                       if e["actual_min_boom_tip_range_m"] >= 0]
        m3b_dwells = [e["actual_M3b_dwell_s"] for e in episodes_data]
        results_json = {
            "checkpoint": os.path.basename(args.checkpoint),
            "stage": args.stage,
            "n_episodes": n,
            "timestamp": datetime.now().isoformat(),
            "m3b_rate": round(sum(1 for e in episodes_data if e["actual_M3b_achieved"]) / n, 4) if n else 0,
            "m3b_count": sum(1 for e in episodes_data if e["actual_M3b_achieved"]),
            "mean_m3b_dwell_s": round(float(np.mean(m3b_dwells)), 4) if m3b_dwells else 0,
            "mean_proximity_m": round(float(np.mean(boom_ranges)), 4) if boom_ranges else None,
            "mean_reward": round(float(np.mean([e["total_reward"] for e in episodes_data])), 4) if n else 0,
            "verdict": verdict,
            "episodes": episodes_data,
        }
        json_path = os.path.join(args.output_dir, f"results_eid{args.env_id}.json")
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2, default=str)

        logger.info("CSV: %s (%d episodes)", csv_path, n)
        logger.info("Summary: %s", summary_path)
        logger.info("JSON: %s", json_path)
        logger.info("VERDICT: %s (M3b %.1f%%)", verdict, m3b_pct)


if __name__ == "__main__":
    main()
