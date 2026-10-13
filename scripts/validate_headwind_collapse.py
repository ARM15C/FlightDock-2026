#!/usr/bin/env python3
"""Fixed 1 m/s headwind validation — confirms baseline collapse at minimal wind.

Runs 3 episodes with constant headwind (no randomization, no turbulence).
Used for defense documentation: even 1 m/s is sufficient to trigger collapse.

Usage:
    python scripts/validate_headwind_collapse.py
"""

from __future__ import annotations

import csv
import logging
import os
import signal
import sys

import numpy as np

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

_shutdown = False

def _signal_handler(signum, frame):
    global _shutdown
    logger.warning("Signal %d — finishing current episode then exiting", signum)
    _shutdown = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def main():
    from drl import config
    from drl.curriculum import CurriculumManager
    from drl.env_formation_sitl import FormationSITLEnv

    N_EPISODES = 3
    ENV_ID = 31  # Avoid collision with eval script's default 30
    SPEEDUP = 2.0
    STAGE = 1

    # Fixed wind override: 1 m/s headwind (dir=0), no turbulence
    FIXED_WIND_SPEED = 1.0
    FIXED_WIND_DIR = 0.0  # wind from north (headwind on straight legs, crosswind during turns)
    FIXED_WIND_TURB = 0.0

    output_dir = os.path.join(_REPO_ROOT, "logs/headwind_validation_1ms")
    os.makedirs(output_dir, exist_ok=True)

    # Increase warmup tolerance
    config.WARMUP_TIMEOUT_S = max(config.WARMUP_TIMEOUT_S, 1200.0 / SPEEDUP * 1.2)
    config.MAX_WARMUP_RETRIES = 5

    curriculum = CurriculumManager(initial_stage=STAGE)

    # Monkeypatch: override domain randomization to return fixed wind
    _original_sample = curriculum.sample_domain_randomization

    def _fixed_sample():
        result = _original_sample()
        result["wind_speed_ms"] = FIXED_WIND_SPEED
        result["wind_dir_deg"] = FIXED_WIND_DIR
        result["wind_turb"] = FIXED_WIND_TURB
        return result

    curriculum.sample_domain_randomization = _fixed_sample

    logger.info(
        "Fixed headwind validation: %.1f m/s headwind, 0 turbulence, %d episodes",
        FIXED_WIND_SPEED, N_EPISODES,
    )

    env = FormationSITLEnv(
        env_id=ENV_ID,
        speedup=SPEEDUP,
        curriculum_manager=curriculum,
        ui_mode="none",
        warmup_max_sim_s=1200.0,
    )
    env.set_safety_range(STAGE)

    CSV_COLUMNS = [
        "episode", "steps", "total_reward", "terminal_type", "fsm_state",
        "actual_M2_achieved", "actual_M2_dwell_s",
        "actual_M3b_achieved", "actual_M3b_dwell_s",
        "min_range_m", "max_behind_overshoot_m", "max_cross_excursion_m",
    ]

    episodes_data = []
    csv_path = os.path.join(output_dir, "metrics.csv")

    try:
        ep_num = 0
        consecutive_failures = 0
        while ep_num < N_EPISODES:
            if _shutdown:
                break

            try:
                obs, info = env.reset()
            except RuntimeError as e:
                consecutive_failures += 1
                logger.warning("Warmup failure %d: %s", consecutive_failures, e)
                if consecutive_failures >= 10:
                    logger.error("Aborting: %d consecutive warmup failures", consecutive_failures)
                    break
                continue

            consecutive_failures = 0
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

            row = {
                "episode": ep_num,
                "steps": step_count,
                "total_reward": round(total_reward, 4),
                "terminal_type": info.get("terminal_type", "unknown"),
                "fsm_state": info.get("fsm_state", "UNKNOWN"),
                "actual_M2_achieved": info.get("actual_M2_achieved", False),
                "actual_M2_dwell_s": round(info.get("actual_M2_dwell_s", 0.0), 3),
                "actual_M3b_achieved": info.get("actual_M3b_achieved", False),
                "actual_M3b_dwell_s": round(info.get("actual_M3b_dwell_s", 0.0), 3),
                "min_range_m": round(info.get("min_range_m", 0.0), 3),
                "max_behind_overshoot_m": round(info.get("max_behind_overshoot_m", 0.0), 3),
                "max_cross_excursion_m": round(info.get("max_cross_excursion_m", 0.0), 3),
            }
            episodes_data.append(row)

            logger.info(
                "Episode %d/%d: %d steps, reward=%.1f, %s/%s, "
                "M3b=%s, range=%.2fm, cross=%.1fm",
                ep_num, N_EPISODES, step_count, total_reward,
                row["terminal_type"], row["fsm_state"],
                row["actual_M3b_achieved"], row["min_range_m"],
                row["max_cross_excursion_m"],
            )

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(episodes_data)

    finally:
        env.close()

    # Summary
    if episodes_data:
        m3b_count = sum(1 for e in episodes_data if e["actual_M3b_achieved"])
        cross_vals = [e["max_cross_excursion_m"] for e in episodes_data]
        range_vals = [e["min_range_m"] for e in episodes_data]
        logger.info("=" * 60)
        logger.info("HEADWIND VALIDATION RESULT (1 m/s, 0 turbulence)")
        logger.info("Episodes: %d", len(episodes_data))
        logger.info("M3b: %d/%d (%.1f%%)", m3b_count, len(episodes_data),
                     100 * m3b_count / len(episodes_data))
        logger.info("Min range: %.2f–%.2fm", min(range_vals), max(range_vals))
        logger.info("Cross excursion: %.1f–%.1fm", min(cross_vals), max(cross_vals))
        logger.info("Collapse confirmed: %s",
                     "YES" if m3b_count == 0 else "NO — unexpected M3b achievement")
        logger.info("=" * 60)
        logger.info("CSV: %s", csv_path)


if __name__ == "__main__":
    main()
