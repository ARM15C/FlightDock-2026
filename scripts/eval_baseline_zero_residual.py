#!/usr/bin/env python3
"""Zero-residual baseline evaluation — measures pure baseline performance
under specified curriculum stage conditions (wind, turbulence, GPS noise).

Runs FormationSITLEnv with action=[0,0,0] for every step. No SB3, no
SubprocVecEnv, no VecNormalize — pure Gymnasium API. Outputs episode
metrics CSV and summary statistics.

Usage:
    python scripts/eval_baseline_zero_residual.py \
        --stage 1 --n-episodes 30 --speedup 2.0 --env-id 30
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import signal
import sys
from datetime import datetime

import numpy as np

# Ensure src/ and tests/ are on path (same pattern as collect_pid_demos.py)
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


def _normalize_fd_fw_variant() -> None:
    """Normalize known legacy FD_FW_VARIANT aliases before SITL launch.

    ``tests/start_formation_sitl_fixed.sh`` accepts only ``{legacy, modern}``
    and exits non-zero on any other value, which causes silent SITL non-spawn
    (and the consequent 30 s heartbeat timeouts in the controller's preflight).
    Several historical campaign scripts (e.g.
    ``analysis/v98_ec2_replication_*/scripts/phase7*.sh``) set
    ``FD_FW_VARIANT=baseline``; that one fossilized alias is normalized here
    to ``legacy`` (the documented default). All other invalid values fail fast
    with an explicit ``ValueError`` to avoid masking real configuration errors.

    Port arithmetic and SITL launch logic are unchanged by this helper.
    """
    valid = {"legacy", "modern"}
    raw = os.environ.get("FD_FW_VARIANT", "")
    norm = raw.strip().lower()
    if not norm:
        os.environ["FD_FW_VARIANT"] = "legacy"
        logger.info("FD_FW_VARIANT unset; defaulting to 'legacy'.")
        return
    if norm == "baseline":
        os.environ["FD_FW_VARIANT"] = "legacy"
        logger.warning(
            "FD_FW_VARIANT='baseline' is a deprecated alias; "
            "normalizing to 'legacy' for start_formation_sitl_fixed.sh."
        )
        return
    if norm in valid:
        os.environ["FD_FW_VARIANT"] = norm
        return
    raise ValueError(
        f"Invalid FD_FW_VARIANT={raw!r}; valid values are "
        "'legacy', 'modern', or deprecated alias 'baseline'."
    )


# Graceful shutdown flag
_shutdown = False


def _signal_handler(signum, frame):
    global _shutdown
    logger.warning("Signal %d received — finishing current episode then exiting", signum)
    _shutdown = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# CSV column order
CSV_COLUMNS = [
    "episode", "steps", "total_reward", "terminal_type", "fsm_state",
    "leader_profile",
    "actual_M2_achieved", "actual_M2_dwell_s",
    "actual_M3a_coarse_achieved", "actual_M3a_coarse_dwell_s",
    "actual_M3a_fine_achieved", "actual_M3a_fine_dwell_s",
    "actual_M3b_achieved", "actual_M3b_dwell_s",
    "min_range_m", "actual_min_boom_tip_range_m",
    "dwell_M2_s", "dwell_M3a_coarse_s", "dwell_M3a_fine_s", "dwell_M3b_s",
    "time_to_M2_s", "time_to_M3a_coarse_s", "time_to_M3a_fine_s", "time_to_M3b_s",
    "max_behind_overshoot_m", "max_cross_excursion_m",
    # C3j Phase 0 diagnostics
    "m3b_entry_count", "m3b_break_count", "m3b_mean_dwell_before_break_s",
    "m3b_breaks_behind", "m3b_breaks_cross", "m3b_breaks_below",
    "behind_axis_exit_count", "non_behind_exit_count",
    "mean_ttc_at_m3b_entry", "mean_closure_rate_delta_in_m3b",
    # T-393/T-397 condition witness (appended -- pre-existing column order is
    # untouched). ARM-B legs were recording no independent variables at all,
    # so an ARM-B row could not be told apart from a calm-air row, and the two
    # benchmark arms had divergent metrics schemas. The legacy `wind_turb`
    # knob is deliberately NOT a column: it is INERT on the json backend
    # (curriculum.py:412-414) and reads 0.0 on genuinely turbulent cells.
    "wind_speed_ms", "wind_from_deg", "turbulence_sigma_ms", "gps_cep_m",
    "v3_cell_label",
]


def _extract_episode_metrics(ep_num, step_count, total_reward, info):
    """Extract flat dict from terminal info for CSV row."""
    return {
        "episode": ep_num,
        "steps": step_count,
        "total_reward": round(total_reward, 4),
        "terminal_type": info.get("terminal_type", "unknown"),
        "fsm_state": info.get("fsm_state", "UNKNOWN"),
        "leader_profile": info.get("leader_profile", "racetrack"),
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
        # C3j Phase 0 diagnostics
        "m3b_entry_count": info.get("m3b_entry_count", 0),
        "m3b_break_count": info.get("m3b_break_count", 0),
        "m3b_mean_dwell_before_break_s": round(info.get("m3b_mean_dwell_before_break_s", 0.0), 4),
        "m3b_breaks_behind": info.get("m3b_breaks_behind", 0),
        "m3b_breaks_cross": info.get("m3b_breaks_cross", 0),
        "m3b_breaks_below": info.get("m3b_breaks_below", 0),
        "behind_axis_exit_count": info.get("behind_axis_exit_count", 0),
        "non_behind_exit_count": info.get("non_behind_exit_count", 0),
        "mean_ttc_at_m3b_entry": round(info.get("mean_ttc_at_m3b_entry", 0.0), 4),
        "mean_closure_rate_delta_in_m3b": round(info.get("mean_closure_rate_delta_in_m3b", 0.0), 4),
        # T-393/T-397: UNSAMPLED renders as "" and never as 0.0 -- a zero
        # reads as "calm air", which is a claim; blank reads as "unknown",
        # which is the truth.
        "wind_speed_ms": info.get("wind_speed_ms", ""),
        "wind_from_deg": info.get("wind_dir_deg", ""),
        "turbulence_sigma_ms": ";".join(
            f"{float(s):.3f}" for s in (info.get("wind_sigma_ms") or [])),
        "gps_cep_m": info.get("gps_accuracy", ""),
        "v3_cell_label": info.get("v3_cell_label", ""),
    }


def _print_summary(episodes_data, stage):
    """Compute and print aggregate statistics."""
    n = len(episodes_data)
    if n == 0:
        logger.warning("No episodes completed — no summary to print.")
        return ""

    lines = []
    lines.append("=" * 70)
    lines.append(f"ZERO-RESIDUAL BASELINE EVALUATION — STAGE {stage}")
    lines.append(f"Episodes completed: {n}")
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
    lines.append(f"  Stage {stage} result: {100*m3b_rate:.1f}% M3b, "
                 f"{np.mean(m3b_dwells_all):.3f}s mean dwell, "
                 f"{np.mean(boom_ranges):.3f}m mean proximity" if boom_ranges else
                 f"  Stage {stage} result: {100*m3b_rate:.1f}% M3b, "
                 f"{np.mean(m3b_dwells_all):.3f}s mean dwell, N/A proximity")

    # 9. C3j Phase 0: Dwell-before-break and corridor dynamics
    dwell_bb = [e["m3b_mean_dwell_before_break_s"] for e in episodes_data
                if e["m3b_break_count"] > 0]
    entries_ep = [e["m3b_entry_count"] for e in episodes_data]
    breaks_behind = sum(e["m3b_breaks_behind"] for e in episodes_data)
    breaks_cross = sum(e["m3b_breaks_cross"] for e in episodes_data)
    breaks_below = sum(e["m3b_breaks_below"] for e in episodes_data)
    total_breaks = breaks_behind + breaks_cross + breaks_below

    lines.append("\n--- C3j Phase 0: Corridor Dynamics ---")
    if dwell_bb:
        lines.append(f"  Dwell-before-break: mean={np.mean(dwell_bb):.4f}s, "
                     f"std={np.std(dwell_bb):.4f}s, "
                     f"min={np.min(dwell_bb):.4f}s, max={np.max(dwell_bb):.4f}s")
    else:
        lines.append("  Dwell-before-break: N/A (no M3b breaks recorded)")
    lines.append(f"  Entries/ep: mean={np.mean(entries_ep):.1f}, "
                 f"std={np.std(entries_ep):.1f}, "
                 f"min={np.min(entries_ep)}, max={np.max(entries_ep)}")
    if total_breaks > 0:
        lines.append(f"  Break axis distribution: behind={breaks_behind} "
                     f"({100*breaks_behind/total_breaks:.1f}%), "
                     f"cross={breaks_cross} ({100*breaks_cross/total_breaks:.1f}%), "
                     f"below={breaks_below} ({100*breaks_below/total_breaks:.1f}%)")
        lines.append(f"  Behind-axis break fraction: {100*breaks_behind/total_breaks:.1f}%")
    else:
        lines.append("  Break axis distribution: N/A (no breaks)")

    # Behind-axis exit penalty diagnostics
    bae = sum(e["behind_axis_exit_count"] for e in episodes_data)
    nbe = sum(e["non_behind_exit_count"] for e in episodes_data)
    lines.append(f"  Exit penalty: behind_axis={bae}, non_behind={nbe}")

    # TTC and closure rate delta
    ttc_vals = [e["mean_ttc_at_m3b_entry"] for e in episodes_data
                if e["m3b_entry_count"] > 0]
    crd_vals = [e["mean_closure_rate_delta_in_m3b"] for e in episodes_data
                if e["m3b_entry_count"] > 0]
    if ttc_vals:
        lines.append(f"  Mean TTC at M3b entry: {np.mean(ttc_vals):.4f}s")
    if crd_vals:
        lines.append(f"  Mean closure_rate_delta in M3b: {np.mean(crd_vals):.4f} m/s^2")

    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    print(summary_text)
    return summary_text


def main():
    parser = argparse.ArgumentParser(
        description="Zero-residual baseline evaluation — pure action=[0,0,0] via Gymnasium API",
    )
    parser.add_argument("--stage", type=int, default=1,
                        help="Curriculum stage (determines wind/turb/GPS bounds)")
    parser.add_argument("--n-episodes", type=int, default=30,
                        help="Number of evaluation episodes")
    parser.add_argument("--speedup", type=float, default=2.0,
                        help="SITL speedup factor")
    parser.add_argument("--env-id", type=int, default=30,
                        help="Port isolation (avoid training envs 0-23)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: logs/baseline_eval_stage{stage})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible domain randomization")
    parser.add_argument("--warmup-sim-s", type=float, default=1200.0,
                        help="Max sim-time for warmup phase (default: 1200s)")
    parser.add_argument("--force-trajectory", type=str, default=None,
                        choices=["racetrack", "straight", "circle"],
                        help="Override curriculum trajectory sampling for "
                             "the entire eval run (F.2 v2 per-trajectory "
                             "forcing).")
    parser.add_argument("--force-circle-omega", type=float, default=None,
                        help="Yaw rate (deg/s) for forced circle. Required "
                             "iff --force-trajectory=circle; disallowed "
                             "otherwise.")
    parser.add_argument("--settle-seconds", type=float, default=30.0,
                        help="Post-SITL-spawn settle delay before baseline "
                             "preflight queries home altitude. Gives ArduPilot "
                             "EKF time to set origin + field elevation, "
                             "avoiding the cold-start race where "
                             "GLOBAL_POSITION_INT.alt returns ~0m instead of "
                             "home MSL. Default 30s; set 0 to disable.")
    args = parser.parse_args()

    if args.force_trajectory == "circle" and args.force_circle_omega is None:
        parser.error("--force-trajectory=circle requires "
                     "--force-circle-omega")
    if args.force_trajectory != "circle" and args.force_circle_omega is not None:
        parser.error("--force-circle-omega is only valid with "
                     "--force-trajectory=circle")

    # FIXAWIRE-719: assert the EVAL
    # contract here — AFTER argparse validation, so a CLI misuse still reports
    # as a CLI misuse, and BEFORE `drl.env_formation_sitl` is imported below,
    # because _FD_RAW_ALIGN_CENTER is read at MODULE IMPORT and a later check
    # would judge an already-baked value. This is the ARM-B baseline harness;
    # without it ARM-B and ARM-F fly DIFFERENT truth substrates and H2
    # (learned vs PID on M3b dwell) is not measurable.
    from drl import env_contract          # 'drl', not 'src.drl': this file
    env_contract.assert_contract(env_contract.EVAL)   # puts src/ on sys.path

    # Normalize fossilized FD_FW_VARIANT alias before SITL launch. See
    # _normalize_fd_fw_variant() docstring above for the failure mode this
    # protects against.
    _normalize_fd_fw_variant()

    if args.output_dir is None:
        args.output_dir = os.path.join(_REPO_ROOT, f"logs/baseline_eval_stage{args.stage}")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Output directory: %s", args.output_dir)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        logger.info("Random seed: %d (random + numpy)", args.seed)

    # Deferred imports (after sys.path setup)
    from drl import config
    from drl.curriculum import CurriculumManager
    from drl.env_formation_sitl import FormationSITLEnv

    # 0. Scale wall-clock warmup timeout to match sim-time limit
    #    At 2x speedup, 1200s sim ≈ 650s wall. Add 20% margin.
    config.WARMUP_TIMEOUT_S = max(
        config.WARMUP_TIMEOUT_S,
        args.warmup_sim_s / args.speedup * 1.2,
    )
    # Increase per-episode warmup retries (warmup is marginal at ~50% success)
    config.MAX_WARMUP_RETRIES = 5
    logger.info("Warmup limits: sim=%.0fs, wall=%.0fs, retries=%d",
                args.warmup_sim_s, config.WARMUP_TIMEOUT_S, config.MAX_WARMUP_RETRIES)

    # 1. Create curriculum manager locked to specified stage
    curriculum = CurriculumManager(initial_stage=args.stage)
    # T-393/T-397: report the REAL turbulence. The legacy `wind_turb` knob is
    # INERT on the json backend (curriculum.py:412-414) and reads 0.0 on cells
    # that are genuinely turbulent -- printing it as "turb=" told operators
    # these legs flew in still air. The flown turbulence is the FDM Dryden
    # sigma. Same wording as eval_drl_checkpoint.py so the two benchmark arms
    # report their conditions identically.
    _st = config.CURRICULUM_STAGES[args.stage]
    _sig = getattr(_st, "wind_sigma_ms", ()) or ()
    logger.info(
        "Curriculum stage: %d (%s) wind_max=%.1f m/s from %.0f deg, "
        "turbulence_sigma_ms=%s, gps_cep_m=%.2f  [legacy SIM_WIND_TURB knob="
        "%.1f, INERT on json backend -- not the turbulence]",
        args.stage, getattr(config, "STAGE_LABELS_V3", {}).get(args.stage, "?"),
        _st.wind_max_ms, getattr(_st, "wind_from_deg", 0.0),
        ",".join(f"{s:.3f}" for s in _sig) if _sig else "NONE",
        _st.gps_accuracy, _st.wind_turb)

    # 1a. Apply instance-level per-trajectory forcing override for F.2 v2.
    # Mirrors the validate_headwind_collapse.py:69-78 precedent: patch the
    # bound method on the single `curriculum` instance shared with the env
    # (FormationSITLEnv stores the reference at self.curriculum_manager;
    # no second CurriculumManager is constructed inside the env).
    if args.force_trajectory is not None:
        _original_sampler = curriculum.sample_domain_randomization

        def _forced_sampler():
            dr = _original_sampler()
            dr["leader_profile"] = args.force_trajectory
            dr["leader_omega_deg_s"] = (
                args.force_circle_omega
                if args.force_trajectory == "circle"
                else None
            )
            return dr

        curriculum.sample_domain_randomization = _forced_sampler
        logger.info("Trajectory forcing enabled: leader_profile=%s, "
                    "leader_omega_deg_s=%s",
                    args.force_trajectory,
                    args.force_circle_omega
                    if args.force_trajectory == "circle" else "None")

    # 2. Create environment
    env = FormationSITLEnv(
        env_id=args.env_id,
        speedup=args.speedup,
        curriculum_manager=curriculum,
        ui_mode="none",
        warmup_max_sim_s=args.warmup_sim_s,
        # T-400: the V3 per-episode RNG derives from the CONSTRUCTOR seed
        # (env:1131 gates on _initial_seed, set only at :646); the reset(seed=)
        # below feeds Gymnasium np_random and NEVER reaches it. Without this
        # ARM-B cannot fly ANY V3 cell: curriculum.py:341 refuses the
        # nondeterministic global-random path, the warmup retries 10x and the
        # leg dies in under a second (T-399 smoke, seed 90027). Same wiring and
        # rationale as the policy harness (eval_drl_checkpoint.py:409-412) --
        # the two benchmark arms must seed identically or their cells differ.
        # None stays None: non-seeded evals are unchanged.
        seed=args.seed,
        out_dir=args.output_dir,
    )

    # 3. Set safety range for stage (before first reset)
    env.set_safety_range(args.stage)
    logger.info("Safety range: %.0f m (stage %d)", env._safety_range_m, args.stage)

    # 3a. Post-SITL-spawn settle hook. On the direct-eval cold-start path
    # (no pool-warm), baseline preflight [3/7] can query
    # GLOBAL_POSITION_INT.alt before ArduPilot EKF has set origin +
    # field elevation, returning ~0m instead of the 584m home MSL.
    # Downstream: takeoff target computed as 890m AGL (should be 306m),
    # climb exceeds the warmup timeout. This monkey-patch inserts a
    # settle delay between SITL spawn and preflight; scope confined to
    # the eval harness (no controller / env changes).
    if args.settle_seconds > 0:
        import time as _settle_time
        _orig_launch_sitl = env._launch_sitl

        def _launch_sitl_then_settle():
            _orig_launch_sitl()
            logger.info(
                "Post-SITL-spawn settle: sleeping %.1fs for EKF "
                "origin + field-elevation convergence",
                args.settle_seconds,
            )
            _settle_time.sleep(args.settle_seconds)

        env._launch_sitl = _launch_sitl_then_settle

    episodes_data = []
    csv_path = os.path.join(args.output_dir, "metrics.csv")

    warmup_failures = 0
    max_warmup_failures = 10  # Give up after this many consecutive failures

    try:
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

            consecutive_failures = 0  # Reset on success
            done = False
            step_count = 0
            total_reward = 0.0

            while not done:
                action = np.zeros(config.ACTION_DIM, dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1
                total_reward += float(reward)

            ep_num += 1

            # Extract metrics from terminal info
            row = _extract_episode_metrics(ep_num, step_count, total_reward, info)
            episodes_data.append(row)

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

    finally:
        env.close()
        if warmup_failures > 0:
            logger.info("Total warmup failures across run: %d", warmup_failures)

    # Print and save summary
    logger.info("Writing summary to %s", args.output_dir)
    summary_text = _print_summary(episodes_data, args.stage)
    if summary_text:
        summary_path = os.path.join(args.output_dir, f"summary_eid{args.env_id}.txt")
        with open(summary_path, "w") as f:
            f.write(summary_text + "\n")
        logger.info("CSV: %s (%d episodes)", csv_path, len(episodes_data))
        logger.info("Summary: %s", summary_path)


if __name__ == "__main__":
    main()
