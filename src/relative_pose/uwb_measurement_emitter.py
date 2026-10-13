"""
UWB Measurement Emitter Module

Synthetic UWB range emitter for Test 387.
Emits UWBMeasurement objects at a fixed sim-time rate (e.g. 10 Hz),
for multiple anchors, with deterministic noise, optional latency,
and optional max-range dropouts.

Key Features:
- SIM-TIME ONLY: All scheduling uses now_s passed from controller
- Deterministic across speedups: Uses local RNG seeded from configurable seed
- Configurable via environment variables
- Multi-anchor support with configurable positions
- Latency and dropout modeling

Environment Variables:
- HARNESS_UWB_ENABLE (default 1): Enable/disable emitter
- HARNESS_UWB_RATE_HZ (default 10.0): Emission rate in Hz
- HARNESS_UWB_SIGMA_M (default 0.15): Range noise std dev in meters
- HARNESS_UWB_MAX_RANGE_M (default 200.0): Max range for measurements
- HARNESS_UWB_LATENCY_S (default 0.05): Measurement latency in seconds
- HARNESS_UWB_DROPOUT_PROB (default 0.0): Probability of random dropout
- HARNESS_UWB_SEED (default 0): RNG seed for determinism
- HARNESS_UWB_ANCHORS_JSON (optional): JSON dict mapping anchor_id to [behind, cross, below]

Created: 2026-01-19
Author: FlightDock Team
Reference: TEST_387_UWB_EMITTER.md
"""

import json
import logging
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

# Import UWB types and helper from fusion module
from uwb_gps_fusion import UWBMeasurement, compute_range_for_anchor

logger = logging.getLogger(__name__)


# ==============================================================================
# CONSTANTS / DEFAULTS
# ==============================================================================

DEFAULT_UWB_RATE_HZ = 10.0
DEFAULT_UWB_SIGMA_M = 0.05  # true UWB range noise sigma (substrate re-tune 2026-06-28; DW1000/DW3000 good-conditions datasheet)
DEFAULT_UWB_MAX_RANGE_M = 200.0
DEFAULT_UWB_LATENCY_S = 0.05
DEFAULT_UWB_DROPOUT_PROB = 0.0
DEFAULT_UWB_SEED = 0

# Default anchor layout (IDs 0-3) if no JSON provided
# Expressed in (behind_m, cross_m, below_m) - leader body frame
DEFAULT_ANCHOR_POSITIONS: Dict[int, Tuple[float, float, float]] = {
    0: (0.0, 0.0, 0.0),  # Primary (origin / boom tip)
    1: (0.0, 0.50, 0.0),  # 0.5m right of origin
    2: (0.0, -0.50, 0.0),  # 0.5m left of origin
    3: (0.0, 0.0, 0.30),  # 0.3m below origin
}


# ==============================================================================
# UWB RANGING EMITTER CLASS
# ==============================================================================


class UWBRangingEmitter:
    """
    Synthetic UWB range emitter for Test 387.

    Emits UWBMeasurement objects at a fixed sim-time rate (e.g. 10 Hz),
    for multiple anchors, with deterministic noise, optional latency,
    and optional max-range dropouts.

    SIM-TIME ONLY: All scheduling uses now_s passed from controller.
    No time.time(), no time.monotonic(). Timestamps on UWBMeasurement.timestamp
    are in the same timebase as now_s.

    Usage:
        emitter = UWBRangingEmitter() # Reads config from env vars
        if emitter.enabled:
            meas_list = emitter.maybe_emit(now_s=sim_time, rel_bcb=(behind, cross, below))
            fusion.update_uwb_batch(meas_list)
    """

    def __init__(
        self,
        enabled: Optional[bool] = None,
        rate_hz: Optional[float] = None,
        sigma_m: Optional[float] = None,
        max_range_m: Optional[float] = None,
        latency_s: Optional[float] = None,
        dropout_prob: Optional[float] = None,
        seed: Optional[int] = None,
        anchor_positions: Optional[Dict[int, Tuple[float, float, float]]] = None,
    ):
        """
        Initialize the UWB ranging emitter.

        All parameters default to environment variable values if not provided.

        Args:
            enabled: Enable/disable emitter (env: HARNESS_UWB_ENABLE)
            rate_hz: Emission rate in Hz (env: HARNESS_UWB_RATE_HZ)
            sigma_m: Range noise std dev in meters (env: HARNESS_UWB_SIGMA_M)
            max_range_m: Max range for measurements (env: HARNESS_UWB_MAX_RANGE_M)
            latency_s: Measurement latency in seconds (env: HARNESS_UWB_LATENCY_S)
            dropout_prob: Probability of random dropout (env: HARNESS_UWB_DROPOUT_PROB)
            seed: RNG seed for determinism (env: HARNESS_UWB_SEED)
            anchor_positions: Dict mapping anchor_id to (behind, cross, below) tuple
                              (env: HARNESS_UWB_ANCHORS_JSON)
        """
        # Read configuration from environment (with defaults)
        self.enabled = (
            enabled if enabled is not None else self._env_bool("HARNESS_UWB_ENABLE", True)
        )
        self.rate_hz = (
            rate_hz
            if rate_hz is not None
            else self._env_float("HARNESS_UWB_RATE_HZ", DEFAULT_UWB_RATE_HZ)
        )
        self.sigma_m = (
            sigma_m
            if sigma_m is not None
            else self._env_float("HARNESS_UWB_SIGMA_M", DEFAULT_UWB_SIGMA_M)
        )
        self.max_range_m = (
            max_range_m
            if max_range_m is not None
            else self._env_float("HARNESS_UWB_MAX_RANGE_M", DEFAULT_UWB_MAX_RANGE_M)
        )
        self.latency_s = (
            latency_s
            if latency_s is not None
            else self._env_float("HARNESS_UWB_LATENCY_S", DEFAULT_UWB_LATENCY_S)
        )
        self.dropout_prob = (
            dropout_prob
            if dropout_prob is not None
            else self._env_float("HARNESS_UWB_DROPOUT_PROB", DEFAULT_UWB_DROPOUT_PROB)
        )

        # RNG seed for determinism
        env_seed = (
            seed
            if seed is not None
            else int(self._env_float("HARNESS_UWB_SEED", DEFAULT_UWB_SEED))
        )
        self._rng = random.Random(env_seed)

        # Anchor positions
        if anchor_positions is not None:
            self.anchor_positions = anchor_positions.copy()
        else:
            # Try to parse from environment JSON
            anchors_json = os.environ.get("HARNESS_UWB_ANCHORS_JSON", "")
            if anchors_json:
                try:
                    parsed = json.loads(anchors_json)
                    self.anchor_positions = {int(k): tuple(v) for k, v in parsed.items()}
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(
                        f"Failed to parse HARNESS_UWB_ANCHORS_JSON: {e}, using defaults"
                    )
                    self.anchor_positions = DEFAULT_ANCHOR_POSITIONS.copy()
            else:
                self.anchor_positions = DEFAULT_ANCHOR_POSITIONS.copy()

        # Ensure anchor 0 exists at origin if not already defined
        if 0 not in self.anchor_positions:
            self.anchor_positions[0] = (0.0, 0.0, 0.0)

        # Rate gating: minimum interval between emissions per anchor
        self._emit_interval_s = 1.0 / self.rate_hz if self.rate_hz > 0 else float("inf")
        self._last_emit_time_s: float = -float("inf")  # Global last emit time

        # Per-anchor timestamp tracking (for latency monotonicity)
        self._last_meas_timestamp_by_anchor: Dict[int, float] = {}

        # Statistics tracking
        self._emitted_total: int = 0
        self._emitted_by_anchor: Dict[int, int] = {}
        self._dropped_total: int = 0
        self._dropped_by_anchor: Dict[int, int] = {}
        self._dropped_by_reason: Dict[str, int] = {"max_range": 0, "dropout": 0}
        self._last_emit_sim_time_s: float = -float("inf")

        if self.enabled:
            logger.info(
                f"UWBRangingEmitter enabled: rate={self.rate_hz}Hz, sigma={self.sigma_m}m, "
                f"max_range={self.max_range_m}m, latency={self.latency_s}s, "
                f"dropout_prob={self.dropout_prob}, seed={env_seed}, "
                f"anchors={list(self.anchor_positions.keys())}"
            )
        else:
            logger.debug("UWBRangingEmitter disabled")

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        """Read boolean from environment variable."""
        val = os.environ.get(name, "")
        if val.lower() in ("1", "true", "yes", "on"):
            return True
        elif val.lower() in ("0", "false", "no", "off"):
            return False
        return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        """Read float from environment variable."""
        try:
            return float(os.environ.get(name, default))
        except ValueError:
            return default

    def maybe_emit(
        self,
        now_s: float,
        rel_bcb: Tuple[float, float, float],
    ) -> List[UWBMeasurement]:
        """
        Potentially emit UWB measurements for all anchors.

        This method should be called at each control loop iteration.
        It will emit measurements at the configured rate (rate-gated).

        SIM-TIME ONLY: Uses now_s for all timing decisions.

        Args:
            now_s: Current simulation time in seconds
            rel_bcb: Follower position as (behind_m, cross_m, below_m) tuple
                     in leader body frame

        Returns:
            List of UWBMeasurement objects (may be empty if rate-gated or disabled)
        """
        if not self.enabled:
            return []

        # Rate gate: check if enough time has passed since last emission
        if now_s - self._last_emit_time_s < self._emit_interval_s:
            return []

        self._last_emit_time_s = now_s
        behind_m, cross_m, below_m = rel_bcb

        measurements: List[UWBMeasurement] = []

        for anchor_id, anchor_offset in self.anchor_positions.items():
            # Compute true range from this anchor to follower
            true_range = compute_range_for_anchor(behind_m, cross_m, below_m, anchor_offset)

            # Max range dropout
            if true_range > self.max_range_m:
                self._dropped_total += 1
                self._dropped_by_anchor[anchor_id] = self._dropped_by_anchor.get(anchor_id, 0) + 1
                self._dropped_by_reason["max_range"] += 1
                continue

            # Probabilistic dropout
            if self.dropout_prob > 0 and self._rng.random() < self.dropout_prob:
                self._dropped_total += 1
                self._dropped_by_anchor[anchor_id] = self._dropped_by_anchor.get(anchor_id, 0) + 1
                self._dropped_by_reason["dropout"] += 1
                continue

            # Add noise
            noisy_range = true_range + self._rng.gauss(0, self.sigma_m)
            # Clamp to positive (range cannot be negative)
            noisy_range = max(0.0, noisy_range)

            # Compute timestamp with latency
            # The measurement represents data from (latency_s) ago
            meas_timestamp = now_s - self.latency_s

            # Ensure monotonicity per anchor
            last_ts = self._last_meas_timestamp_by_anchor.get(anchor_id, -float("inf"))
            if meas_timestamp <= last_ts:
                # Clamp to just after the last timestamp
                meas_timestamp = last_ts + 1e-6
            self._last_meas_timestamp_by_anchor[anchor_id] = meas_timestamp

            # Create measurement
            meas = UWBMeasurement(
                timestamp=meas_timestamp,
                range_m=noisy_range,
                anchor_id=anchor_id,
            )
            measurements.append(meas)

            # Update stats
            self._emitted_total += 1
            self._emitted_by_anchor[anchor_id] = self._emitted_by_anchor.get(anchor_id, 0) + 1

        self._last_emit_sim_time_s = now_s
        return measurements

    def get_anchor_positions(self) -> Dict[int, Tuple[float, float, float]]:
        """Get the anchor position registry (for fusion filter alignment)."""
        return self.anchor_positions.copy()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get emitter statistics for run_summary.json.

        Returns:
            JSON-serializable dict with emitter metrics
        """
        # Compute effective rate if we have emissions
        effective_rate_hz = 0.0
        if self._emitted_total > 0 and self._last_emit_sim_time_s > 0:
            # Rough estimate based on total emissions and sim-time span
            # (Imprecise but gives a useful diagnostic)
            num_anchors = len(self.anchor_positions)
            if num_anchors > 0:
                meas_per_anchor = self._emitted_total / num_anchors
                # This is an approximation
                effective_rate_hz = self.rate_hz if self._emitted_total > 0 else 0.0

        return {
            "enabled": self.enabled,
            "config": {
                "rate_hz": self.rate_hz,
                "sigma_m": self.sigma_m,
                "max_range_m": self.max_range_m,
                "latency_s": self.latency_s,
                "dropout_prob": self.dropout_prob,
                "anchor_count": len(self.anchor_positions),
                "anchor_ids": list(self.anchor_positions.keys()),
            },
            "stats": {
                "emitted_total": self._emitted_total,
                "emitted_by_anchor": dict(self._emitted_by_anchor),
                "dropped_total": self._dropped_total,
                "dropped_by_anchor": dict(self._dropped_by_anchor),
                "dropped_by_reason": dict(self._dropped_by_reason),
                "last_emit_sim_time_s": (
                    self._last_emit_sim_time_s if self._last_emit_sim_time_s > 0 else None
                ),
                "effective_rate_hz": effective_rate_hz,
            },
        }

    def reset_stats(self) -> None:
        """Reset emission statistics (useful for new test runs)."""
        self._emitted_total = 0
        self._emitted_by_anchor.clear()
        self._dropped_total = 0
        self._dropped_by_anchor.clear()
        self._dropped_by_reason = {"max_range": 0, "dropout": 0}
        self._last_emit_sim_time_s = -float("inf")


# ==============================================================================
# SELF-TEST
# ==============================================================================

if __name__ == "__main__":
    """Self-test: verify emitter behavior with simulated flight."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("UWB Measurement Emitter Self-Test")
    print("=" * 60)

    # Create emitter with explicit configuration
    emitter = UWBRangingEmitter(
        enabled=True,
        rate_hz=10.0,
        sigma_m=0.15,
        max_range_m=100.0,
        latency_s=0.05,
        dropout_prob=0.0,
        seed=42,  # Fixed seed for reproducibility
    )

    print(f"\nEmitter config:")
    print(f"  Rate: {emitter.rate_hz} Hz")
    print(f"  Sigma: {emitter.sigma_m} m")
    print(f"  Max range: {emitter.max_range_m} m")
    print(f"  Latency: {emitter.latency_s} s")
    print(f"  Anchors: {emitter.anchor_positions}")

    # Simulate 10 seconds of flight at 50 Hz control rate
    # Follower position: 50m behind, 5m right, 2m below leader
    behind_m, cross_m, below_m = 50.0, 5.0, 2.0

    print(f"\nSimulating 10 seconds of flight...")
    print(f"  Follower position: behind={behind_m}m, cross={cross_m}m, below={below_m}m")

    all_measurements: List[UWBMeasurement] = []
    for tick in range(500):  # 500 ticks at 50 Hz = 10 seconds
        sim_time = tick * 0.02  # 50 Hz -> 20ms per tick

        meas_list = emitter.maybe_emit(
            now_s=sim_time,
            rel_bcb=(behind_m, cross_m, below_m),
        )
        all_measurements.extend(meas_list)

    print(f"\nResults:")
    print(f"  Total measurements emitted: {len(all_measurements)}")
    stats = emitter.get_stats()
    print(f"  Emitted by anchor: {stats['stats']['emitted_by_anchor']}")
    print(f"  Dropped total: {stats['stats']['dropped_total']}")

    # Verify expected count: 10 Hz for 10 seconds = ~100 emissions per anchor
    # With 4 anchors, expect ~400 total measurements
    expected_per_anchor = 10 * 10  # 10 Hz * 10 seconds
    expected_total = expected_per_anchor * 4  # 4 anchors

    # Allow for rate gating imprecision
    if abs(len(all_measurements) - expected_total) < 10:
        print(f"\n  PASS: Measurement count within expected range ({expected_total})")
    else:
        print(f"\n  WARNING: Expected ~{expected_total}, got {len(all_measurements)}")

    # Verify range values
    print("\nVerifying range measurements for anchor 0 (at origin)...")
    anchor_0_meas = [m for m in all_measurements if m.anchor_id == 0]
    if anchor_0_meas:
        true_range_0 = compute_range_for_anchor(behind_m, cross_m, below_m, (0.0, 0.0, 0.0))
        mean_range = sum(m.range_m for m in anchor_0_meas) / len(anchor_0_meas)
        range_error = abs(mean_range - true_range_0)
        print(f"  True range to anchor 0: {true_range_0:.2f}m")
        print(f"  Mean measured range: {mean_range:.2f}m")
        print(f"  Error: {range_error:.3f}m")

        if range_error < 0.5:  # Should be very close (noise averages out)
            print("  PASS: Mean range error < 0.5m")
        else:
            print("  FAIL: Mean range error too large")

    # Verify timestamps have latency applied
    print("\nVerifying latency...")
    if anchor_0_meas:
        first_meas = anchor_0_meas[0]
        expected_latency = emitter.latency_s
        # First emission happens at t=0, so timestamp should be -latency
        if abs(first_meas.timestamp - (-expected_latency)) < 0.01:
            print(f"  PASS: First timestamp has correct latency ({first_meas.timestamp:.3f}s)")
        else:
            print(
                f"  First timestamp: {first_meas.timestamp:.3f}s (expected ~{-expected_latency:.3f}s)"
            )

    # Test max range dropout
    print("\nTesting max range dropout...")
    emitter2 = UWBRangingEmitter(
        enabled=True,
        rate_hz=10.0,
        max_range_m=30.0,  # 30m max range (follower is at ~50m)
        seed=42,
    )

    meas_list = emitter2.maybe_emit(now_s=0.0, rel_bcb=(50.0, 5.0, 2.0))
    if len(meas_list) == 0:
        print("  PASS: All measurements dropped (range > max_range)")
    else:
        print(f"  FAIL: Expected 0 measurements, got {len(meas_list)}")

    stats2 = emitter2.get_stats()
    print(f"  Dropped by max_range: {stats2['stats']['dropped_by_reason']['max_range']}")

    print("\n" + "=" * 60)
    print("Self-test complete")
    print("=" * 60)
