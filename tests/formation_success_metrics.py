"""FlightDock Formation Success Metrics (Simulation-Time Consistent).

Intended usage:
- Instantiate once inside the formation controller.
- Call set_t0() immediately after SimTimeAuthority PASS gate.
- Call update_geometry() once per controller loop with BOTH boom-tip and leader-center geometry.
- At run end, call finalize() then to_dict() and store into run_summary.json under key "success_metrics".

Design constraints:
- All timing and dt come from sim_time_s (MAVLink time_boot_ms-based) only.
- No wall-clock time in any metric logic.

This module is purposely self-contained (no PX4/ROS deps).

CLASS ARCHITECTURE:
    Public API (import these):
    - FormationSuccessMetrics: Main class for collecting and classifying metrics
    - OutcomeClass: Enum of final outcome classifications (SUCCESS_CONTACT, FAIL, etc.)
    - SafetyCode: Enum of safety violation codes (S1_OVERTAKE, S2_COLLISION, etc.)
    - MetricsThresholds: Container for all threshold configuration

    Configuration dataclasses (used via MetricsThresholds):
    - MilestoneThresholds: M1/M2/M3 staging and docking envelopes
    - SafetyThresholds: Overtake, collision, and fail-state definitions
    - StabilityThresholds: Rolling-window RMS/P2P evaluation bounds
    - AltitudeThresholds: Relative and absolute altitude bands
    - PlateauThresholds: Plateau (stuck) detection parameters
    - DtThresholds: dt clamping bounds
    - IntegrityThresholds: DRL training quality gates

    Internal classes (implementation details):
    - MilestoneState: Per-milestone tracking (entries, dwell, achieved)
    - SafetyEvent: Single safety violation record
    - StabilityWindow: Rolling-window statistics calculator
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

# -------------------------
# Git metadata helper
# -------------------------

# Cache directory for git operations (tests directory)
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_git_metadata() -> Dict[str, Any]:
    """Compute git metadata once per instantiation (not per-tick).

    P7.5 v9.7: Returns git commit hash and working tree clean status.
    Called once in __init__ and cached in instance variable.

    Returns:
        Dict with git_commit (str) and working_tree_clean (bool or None).
    """
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=_TESTS_DIR,
            )
            .decode()
            .strip()
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            cwd=_TESTS_DIR,
        ).decode()
        return {
            "git_commit": commit,
            "working_tree_clean": len(status.strip()) == 0,
        }
    except (subprocess.SubprocessError, OSError):
        return {
            "git_commit": "unknown",
            "working_tree_clean": None,
        }


# -------------------------
# Utility helpers
# -------------------------


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _now_rel(sim_time_s: float, t0_s: float) -> float:
    """Return time since t0 (seconds)."""
    return sim_time_s - t0_s


def _geom_is_sane(
    behind_m: float,
    cross_m: float,
    below_m: float,
    range_m: float,
    margin_m: float = 50.0,
) -> bool:
    """Check if geometry components are consistent with range.

    For valid geometry: sqrt(behind^2 + cross^2 + below^2) ≈ range.
    Each component magnitude should satisfy: |component| <= range + margin.

    Returns False if:
    - Any value is NaN or inf
    - range_m < 0
    - Any component magnitude exceeds range + margin
    """
    # Check for NaN/inf
    values = [behind_m, cross_m, below_m, range_m]
    if any(math.isnan(v) or math.isinf(v) for v in values):
        return False
    # range must be non-negative
    if range_m < 0:
        return False
    # Each component magnitude should not exceed range + margin
    limit = range_m + margin_m
    if abs(behind_m) > limit or abs(cross_m) > limit or abs(below_m) > limit:
        return False
    return True


# -------------------------
# Threshold configuration
# -------------------------


@dataclass(frozen=True)
class MilestoneThresholds:
    # M1 staging corridor (loose)
    m1_behind_min: float = 200.0
    m1_behind_max: float = 250.0
    m1_cross_tol: float = 50.0
    m1_below_tol: float = 50.0

    # M2 precontact hold
    m2_behind_target: float = 10.0
    m2_behind_tol: float = 2.0
    m2_cross_tol: float = 2.0
    m2_below_target: float = 10.0
    m2_below_tol: float = 2.0
    m2_dwell_s: float = 3.0
    m2_deadline_s: float = 600.0

    # M3a coarse contact — Tranche D: boom-tip frame (tolerances around nominal tip)
    # Evaluated as: |delta_behind| < tol, |delta_below| < tol, |delta_cross| < tol
    # where delta_X = center_X - BOOM_TIP_NOMINAL_X
    # Equivalent leader CG frame: behind 3.1-5.5m, below 3.1-5.5m, cross +/-1.0m
    m3a_delta_behind_tol: float = 1.2  # |delta_behind| < 1.2m
    m3a_delta_below_tol: float = 1.2  # |delta_below| < 1.2m
    m3a_delta_cross_tol: float = 1.0  # |delta_cross| < 1.0m
    m3a_dwell_s: float = 2.0  # 2.0s dwell (was 1.0s)
    m3a_deadline_s: float = 1200.0

    # M3b tight contact — Tranche D: boom-tip frame
    # Equivalent leader CG frame: behind 3.6-5.0m, below 3.6-5.0m, cross +/-0.5m
    m3b_delta_behind_tol: float = 0.7  # |delta_behind| < 0.7m
    m3b_delta_below_tol: float = 0.7  # |delta_below| < 0.7m
    m3b_delta_cross_tol: float = 0.5  # |delta_cross| < 0.5m
    m3b_dwell_s: float = 3.0  # 3.0s dwell (was 1.0s)

    # M3a_coarse: Acquisition tier (report-only, §3.X.3) — boom-tip frame
    m3a_coarse_delta_behind_tol: float = 2.0
    m3a_coarse_delta_cross_tol: float = 1.5
    m3a_coarse_delta_below_tol: float = 2.0
    m3a_coarse_dwell_s: float = 1.0

    # Legacy fields (kept for backwards compatibility with old configs)
    m3a_behind_min: float = 0.0
    m3a_behind_max: float = 0.75
    m3a_cross_tol: float = 0.75
    m3a_below_tol: float = 0.75
    m3b_behind_min: float = 0.0
    m3b_behind_max: float = 0.30
    m3b_cross_tol: float = 0.35
    m3b_below_tol: float = 0.35
    m3a_coarse_behind_max: float = 1.50
    m3a_coarse_cross_tol: float = 0.75
    m3a_coarse_below_tol: float = 0.75


@dataclass(frozen=True)
class SafetyThresholds:
    # S1 overtake
    overtake_behind_center_m: float = -1.0
    overtake_persist_s: float = 0.10

    # S2 collision — two-tier
    collision_hard_floor_m: float = 1.0  # Tier 1: unconditional hard-fail range
    collision_hard_floor_persist_s: float = 0.05  # Tier 1: persistence
    collision_range_center_m: float = 3.0  # Tier 2: proximity threshold
    collision_closure_rate_mps: float = 1.0  # Tier 2: closure rate gate (m/s, positive = closing)
    collision_persist_s: float = 0.10  # Tier 2: persistence (raised from 0.05)
    near_miss_range_center_m: float = 5.0

    # S3 NaN/Inf persistence
    nan_persist_hard_fail_s: float = 1.0

    # S4 controller fail states
    hard_fail_states: Tuple[str, ...] = ("FAIL", "NO_PASS_ABORT")

    # Geometry sanity
    geom_sanity_margin_m: float = 50.0

    # Safety evaluation gating (split proximity thresholds)
    overtake_eval_max_range_m: float = 200.0  # skip S1 overtake if range > this
    collision_eval_max_range_m: float = 50.0  # skip S2 collision if range > this
    safety_eval_states: Tuple[str, ...] = (
        "REJOIN_PRECONTACT",
        "HOLD_PRECONTACT",
        "ALT_CAP_HOLD_RANGE",  # P1 v9.4
        "ALTITUDE_CAPTURE",
        "DOCKING_CREEP",
        "HOLD_CONTACT",
    )


@dataclass(frozen=True)
class StabilityThresholds:
    stability_window_s: float = 10.0

    # Precontact target (boom-tip frame)
    precontact_target_behind_m: float = 10.0
    precontact_target_cross_m: float = 0.0
    precontact_target_below_m: float = 10.0

    # Contact target (boom-tip frame)
    contact_target_behind_m: float = 0.0
    contact_target_cross_m: float = 0.0
    contact_target_below_m: float = 0.0

    # State names (configurable)
    precontact_states: Tuple[str, ...] = (
        "REJOIN_PRECONTACT",
        "HOLD_PRECONTACT",
        "ALT_CAP_HOLD_RANGE",  # P1 v9.4
        "ALTITUDE_CAPTURE",
    )
    contact_states: Tuple[str, ...] = (
        "DOCKING_CREEP",
        "HOLD_CONTACT",
    )

    # Optional thresholds used only for outcome classification
    precontact_rms_max_m: float = 1.5
    precontact_p2p_cross_max_m: float = 4.0
    precontact_p2p_below_max_m: float = 4.0

    contact_rms_behind_max_m: float = 0.25
    contact_rms_cross_max_m: float = 0.20
    contact_rms_below_max_m: float = 0.50


@dataclass(frozen=True)
class AltitudeThresholds:
    # Default: relative (uses below_tip_m)
    mode: str = "RELATIVE"  # RELATIVE | ABSOLUTE | BOTH

    precontact_below_target_m: float = 10.0
    precontact_below_tol_m: float = 2.0

    contact_below_target_m: float = 0.0
    contact_below_tol_m: float = 0.5

    # Optional absolute bands (scenario-specific)
    precontact_alt_msl_target: Optional[float] = None
    precontact_alt_msl_tol: float = 3.0

    contact_alt_msl_target: Optional[float] = None
    contact_alt_msl_tol: float = 3.0


@dataclass(frozen=True)
class PlateauThresholds:
    plateau_window_s: float = 60.0
    closure_rate_abs_tol_mps: float = 0.10
    behind_min_m: float = 20.0
    # Only evaluate plateau after entering one of these states
    plateau_states: Tuple[str, ...] = (
        "REJOIN_PRECONTACT",
        "HOLD_PRECONTACT",
    )


@dataclass(frozen=True)
class DtThresholds:
    dt_min_s: float = 0.001
    dt_max_s: float = 0.5


@dataclass(frozen=True)
class IntegrityThresholds:
    """Thresholds for run integrity validation (DRL training gates)."""

    min_sample_count: int = 100  # Reject if fewer samples
    min_percent_valid_tip: float = 50.0  # Reject if tip validity < this %
    min_percent_valid_center: float = 50.0  # Reject if center validity < this %
    min_percent_safety_evaluated: float = 10.0  # Warn if safety eval % < this


@dataclass(frozen=True)
class MetricsThresholds:
    milestones: MilestoneThresholds = MilestoneThresholds()
    safety: SafetyThresholds = SafetyThresholds()
    stability: StabilityThresholds = StabilityThresholds()
    altitude: AltitudeThresholds = AltitudeThresholds()
    plateau: PlateauThresholds = PlateauThresholds()
    dt: DtThresholds = DtThresholds()
    integrity: IntegrityThresholds = IntegrityThresholds()


# -------------------------
# Output enums
# -------------------------


class OutcomeClass(str, Enum):
    SUCCESS_CONTACT = "SUCCESS_CONTACT"
    SUCCESS_PRECONTACT_ONLY = "SUCCESS_PRECONTACT_ONLY"
    PLATEAU = "PLATEAU"
    HOLD_OSCILLATION = "HOLD_OSCILLATION"
    OVERSHOOT = "OVERSHOOT"
    COLLISION = "COLLISION"
    CRASH = "CRASH"
    FAIL = "FAIL"
    INCOMPLETE = "INCOMPLETE"


class SafetyCode(str, Enum):
    S1_OVERTAKE = "S1_OVERTAKE"
    S2_COLLISION = "S2_COLLISION"
    S2_NEAR_MISS = "S2_NEAR_MISS"
    S3_NAN_PERSIST = "S3_NAN_PERSIST"
    S3_STATE_FAIL = "S3_STATE_FAIL"
    S4_TELEMETRY_JUMP = "S4_TELEMETRY_JUMP"


class MaxTierAchieved(str, Enum):
    """Ordinal precision ladder."""

    NONE = "NONE"
    ACQUISITION = "ACQUISITION"  # M3a_coarse achieved
    OPERATIONAL = "OPERATIONAL"  # M3a achieved
    STRICT = "STRICT"  # M3b achieved


# -------------------------
# Internal data structures
# -------------------------


@dataclass
class MilestoneState:
    achieved: bool = False
    first_entry_s: Optional[float] = None
    first_dwell_complete_s: Optional[float] = None
    best_dwell_s: float = 0.0
    entries: int = 0

    # integration state
    _current_dwell_s: float = 0.0

    # meta
    deadline_s: Optional[float] = None

    def to_public_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "achieved": bool(self.achieved),
            "first_entry_s": self.first_entry_s,
            "first_dwell_complete_s": self.first_dwell_complete_s,
            "best_dwell_s": float(self.best_dwell_s),
            "entries": int(self.entries),
        }
        if self.deadline_s is not None:
            d["deadline_s"] = float(self.deadline_s)
            d["met_deadline"] = (
                self.first_dwell_complete_s is not None
                and self.first_dwell_complete_s <= self.deadline_s
            )
        else:
            d["deadline_s"] = None
            d["met_deadline"] = None
        return d


@dataclass
class DwellSegment:
    """One contiguous dwell period within a milestone envelope (O-FIX segment proof)."""

    t_start: float
    t_end: float
    duration_s: float
    max_abs_delta_behind: float
    max_abs_delta_below: float
    max_abs_delta_cross: float
    state_at_entry: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_start": round(self.t_start, 3),
            "t_end": round(self.t_end, 3),
            "duration_s": round(self.duration_s, 3),
            "max_abs_delta_behind": round(self.max_abs_delta_behind, 4),
            "max_abs_delta_below": round(self.max_abs_delta_below, 4),
            "max_abs_delta_cross": round(self.max_abs_delta_cross, 4),
            "state_at_entry": self.state_at_entry,
        }


@dataclass
class SafetyEvent:
    code: SafetyCode
    severity: str  # "WARN" | "HARD_FAIL"
    first_time_s: float
    duration_s: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "severity": self.severity,
            "first_time_s": float(self.first_time_s),
            "duration_s": None if self.duration_s is None else float(self.duration_s),
            "details": self.details,
        }


class StabilityWindow:
    """Rolling-window stability calculator for a phase group."""

    def __init__(self, window_s: float) -> None:
        self.window_s = float(window_s)
        self._buf: Deque[Tuple[float, float, float, float]] = deque()  # (t, eB, eC, eZ)
        self.latest: Dict[str, Any] = {
            "window_s": self.window_s,
            "sample_count": 0,
            "rms_behind": None,
            "rms_cross": None,
            "rms_below": None,
            "p2p_behind": None,
            "p2p_cross": None,
            "p2p_below": None,
            "mean_behind": None,
            "mean_cross": None,
            "mean_below": None,
        }

    def add(self, sim_time_s: float, eB: float, eC: float, eZ: float) -> None:
        self._buf.append((sim_time_s, eB, eC, eZ))
        self._trim(sim_time_s)
        self._recompute()

    def _trim(self, sim_time_s: float) -> None:
        cutoff = sim_time_s - self.window_s
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def _recompute(self) -> None:
        n = len(self._buf)
        self.latest["sample_count"] = n
        if n == 0:
            return

        eBs = [x[1] for x in self._buf]
        eCs = [x[2] for x in self._buf]
        eZs = [x[3] for x in self._buf]

        def rms(vals: Sequence[float]) -> float:
            return math.sqrt(sum(v * v for v in vals) / len(vals))

        def mean(vals: Sequence[float]) -> float:
            return sum(vals) / len(vals)

        def p2p(vals: Sequence[float]) -> float:
            return max(vals) - min(vals)

        self.latest.update(
            {
                "rms_behind": rms(eBs),
                "rms_cross": rms(eCs),
                "rms_below": rms(eZs),
                "p2p_behind": p2p(eBs),
                "p2p_cross": p2p(eCs),
                "p2p_below": p2p(eZs),
                "mean_behind": mean(eBs),
                "mean_cross": mean(eCs),
                "mean_below": mean(eZs),
            }
        )


# -------------------------
# Public API
# -------------------------


class FormationSuccessMetrics:
    """Collects and classifies formation success metrics."""

    SPEC_VERSION = "1.0"

    def __init__(
        self,
        boom_length_m: float = 6.1,
        thresholds: Optional[MetricsThresholds] = None,
        thresholds_path: Optional[str] = None,
    ) -> None:
        self.boom_length_m = float(boom_length_m)

        if thresholds is not None:
            self.th = thresholds
        else:
            self.th = (
                self._load_thresholds_from_path(thresholds_path)
                if thresholds_path
                else MetricsThresholds()
            )

        # timebase
        self.t0_s: Optional[float] = None
        self._last_sim_time_s: Optional[float] = None

        # dt stats
        self._dt_values: List[float] = []
        self._sample_count: int = 0
        self._valid_sample_count: int = 0
        self._invalid_sample_count: int = 0

        # invalid streak tracking
        self._invalid_streak_s: float = 0.0
        self._last_valid_time_s: Optional[float] = None

        # range jump tracking
        self._last_range_center_m: Optional[float] = None
        self._max_range_jump_m: Optional[float] = None

        # closure rate tracking (for plateau)
        self._last_behind_tip_m: Optional[float] = None
        self._closure_rate_hist: Deque[Tuple[float, float, float]] = (
            deque()
        )  # (t, closure_rate, behind_tip)

        # plateau accumulator (for continuous persistence detection)
        self._plateau_accum_s: float = 0.0  # Accumulated plateau time
        self._plateau_detected: bool = False  # Latched once threshold reached
        self._plateau_first_detected_s: Optional[float] = None  # Relative to t0

        # milestones
        self.m1 = MilestoneState()
        self.m2 = MilestoneState(deadline_s=self.th.milestones.m2_deadline_s)
        self.m3a = MilestoneState(deadline_s=self.th.milestones.m3a_deadline_s)
        self.m3b = MilestoneState()
        # Step 1a: M3a_coarse_delta — actual coarse thresholds (2.0/2.0/1.5, 1.0s dwell)
        self.m3a_coarse_delta = MilestoneState()

        # safety
        self._overtake_accum_s: float = 0.0
        self._collision_accum_s: float = 0.0  # Tier 2 accumulator
        self._collision_hard_floor_accum_s: float = 0.0  # Tier 1 accumulator
        self._near_miss_seen: bool = False
        self.safety_events: List[SafetyEvent] = []
        self.hard_fail: bool = False
        self.hard_fail_reason: str = ""

        # stability windows
        self._stab_pre = StabilityWindow(self.th.stability.stability_window_s)
        self._stab_con = StabilityWindow(self.th.stability.stability_window_s)

        # altitude violation time
        self.precontact_violation_time_s: float = 0.0
        self.contact_violation_time_s: float = 0.0

        # range tracking stats
        self.min_range_tip_m: Optional[float] = None
        self.min_range_center_m: Optional[float] = None
        self.min_behind_tip_m: Optional[float] = None
        self.min_behind_center_m: Optional[float] = None
        # Timestamps for when minima occurred (relative to t0)
        self.min_range_tip_time_s: Optional[float] = None
        self.min_range_center_time_s: Optional[float] = None
        self.min_behind_tip_time_s: Optional[float] = None
        self.min_behind_center_time_s: Optional[float] = None
        self.max_abs_cross_tip_m: Optional[float] = None
        self.max_abs_below_tip_m: Optional[float] = None
        # P6.2 v9.6: Additional context at min_behind_tip
        self.state_at_min_behind_tip: Optional[str] = None
        self.below_m_at_min_behind_tip: Optional[float] = None
        self.cross_m_at_min_behind_tip: Optional[float] = None

        # Geometry sanity + safety gating tracking
        self._center_insane_count: int = 0
        self._center_insane_first_s: Optional[float] = None
        self._safety_skipped_far_count: int = 0
        self._s1_skipped_state_count: int = 0  # S1-specific state gating (S2 ignores state)
        self._safety_evaluated_count: int = 0
        self._s1_evaluated_count: int = 0
        self._s2_evaluated_count: int = 0

        # Integrity tracking (for DRL training validation)
        self._sample_count_total: int = 0
        self._sample_count_valid_tip: int = 0
        self._sample_count_valid_center: int = 0

        # sim end time
        self.sim_time_end_s: Optional[float] = None

        # outcome
        self.outcome: Optional[OutcomeClass] = None

        # P2 v9.4: Speed arbitration stats (populated by controller)
        self._speed_arb_stats: Optional[Dict[str, Any]] = None
        # Option A+E: DOCKING_CREEP sub-phase stats
        self._dc_phase_stats: Optional[Dict[str, Any]] = None

        # Tracking for min range state context
        self._min_range_tip_state: Optional[str] = None

        # P7.5 v9.7: Git metadata (computed once at init, cached)
        self._git_metadata: Dict[str, Any] = _get_git_metadata()

        # Tranche D: Boom tip nominal position (leader CG frame)
        # Import here to avoid circular dependency; fallback to computed values
        try:
            from formation_config import (
                BOOM_TIP_NOMINAL_BEHIND_M,
                BOOM_TIP_NOMINAL_BELOW_M,
                BOOM_TIP_NOMINAL_CROSS_M,
            )

            self._boom_tip_behind = float(BOOM_TIP_NOMINAL_BEHIND_M)
            self._boom_tip_below = float(BOOM_TIP_NOMINAL_BELOW_M)
            self._boom_tip_cross = float(BOOM_TIP_NOMINAL_CROSS_M)
        except ImportError:
            # Fallback: 45 deg boom, 6.1m extended
            self._boom_tip_behind = 6.1 * math.cos(math.radians(45.0))
            self._boom_tip_below = 6.1 * math.sin(math.radians(45.0))
            self._boom_tip_cross = 0.0

        # Tranche D: Per-tick delta cache (for CSV output)
        self._last_delta_behind: float = 0.0
        self._last_delta_below: float = 0.0
        self._last_delta_cross: float = 0.0

        # Tranche D: M3 dwell tracking (raw + grace)
        self._m3a_dwell_raw_s: float = 0.0
        self._m3a_dwell_grace_s: float = 0.0
        self._m3a_grace_timer_s: float = 0.0
        self._m3b_dwell_raw_s: float = 0.0
        self._m3b_dwell_grace_s: float = 0.0
        self._m3b_grace_timer_s: float = 0.0
        # Step 1a: M3a_coarse_delta dwell tracking
        self._m3a_cd_dwell_raw_s: float = 0.0
        self._m3a_cd_dwell_grace_s: float = 0.0
        self._m3a_cd_grace_timer_s: float = 0.0
        try:
            from formation_config import (
                M3A_DWELL_GRACE_S, M3B_DWELL_GRACE_S,
                FD_M3B_EXCURSION_TOL_ENABLE, M3B_DWELL_EXCURSION_D_S,
                M3B_DWELL_EXCURSION_MAG_M)

            self._m3a_grace_limit_s: float = float(M3A_DWELL_GRACE_S)
            self._m3b_grace_limit_s: float = float(M3B_DWELL_GRACE_S)
            # #460 Candidate B: bounded-excursion tolerance (default-OFF => byte-identical).
            self._m3b_excursion_tol: bool = bool(FD_M3B_EXCURSION_TOL_ENABLE)
            self._m3b_excursion_d_s: float = float(M3B_DWELL_EXCURSION_D_S)
            self._m3b_excursion_mag_m: float = float(M3B_DWELL_EXCURSION_MAG_M)
        except ImportError:
            self._m3a_grace_limit_s = 0.25
            self._m3b_grace_limit_s = 0.25
            self._m3b_excursion_tol = False
            self._m3b_excursion_d_s = 0.5
            self._m3b_excursion_mag_m = 1.0

        # O-FIX: Dwell segment proof tracking (per delta-frame milestone)
        self._m3a_segments: List[DwellSegment] = []
        self._m3a_seg_start: Optional[float] = None
        self._m3a_seg_state: Optional[str] = None
        self._m3a_seg_max_behind: float = 0.0
        self._m3a_seg_max_below: float = 0.0
        self._m3a_seg_max_cross: float = 0.0

        self._m3b_segments: List[DwellSegment] = []
        self._m3b_seg_start: Optional[float] = None
        self._m3b_seg_state: Optional[str] = None
        self._m3b_seg_max_behind: float = 0.0
        self._m3b_seg_max_below: float = 0.0
        self._m3b_seg_max_cross: float = 0.0

        self._m3a_cd_segments: List[DwellSegment] = []
        self._m3a_cd_seg_start: Optional[float] = None
        self._m3a_cd_seg_state: Optional[str] = None
        self._m3a_cd_seg_max_behind: float = 0.0
        self._m3a_cd_seg_max_below: float = 0.0
        self._m3a_cd_seg_max_cross: float = 0.0

        # Rec 5: Sim-time indexed state transition timeline
        # Each entry: {"t_s": float (relative to t0), "from": str, "to": str, "reason": str}
        self._state_timeline: List[Dict[str, Any]] = []

    # -------------------------
    # Property aliases (test compatibility)
    # -------------------------

    @property
    def safety_hard_fail(self) -> bool:
        """Alias for hard_fail (test compatibility)."""
        return self.hard_fail

    # -------------------------
    # Configuration loading
    # -------------------------

    @staticmethod
    def _load_thresholds_from_path(path: str) -> MetricsThresholds:
        """Load thresholds from JSON file. Missing keys fall back to defaults."""
        if not path:
            return MetricsThresholds()
        if not os.path.exists(path):
            raise FileNotFoundError(f"metrics thresholds file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Merge dict -> dataclasses. Keep permissive: ignore unknown keys.
        def merge_dataclass(dc_cls, defaults, dct):
            kwargs = {}
            for k in defaults.__dict__.keys():
                if k in dct:
                    kwargs[k] = dct[k]
            return dc_cls(**kwargs)

        defaults = MetricsThresholds()
        milestones = merge_dataclass(
            MilestoneThresholds, defaults.milestones, data.get("milestones", {})
        )
        safety = merge_dataclass(SafetyThresholds, defaults.safety, data.get("safety", {}))
        stability = merge_dataclass(
            StabilityThresholds, defaults.stability, data.get("stability", {})
        )
        altitude = merge_dataclass(AltitudeThresholds, defaults.altitude, data.get("altitude", {}))
        plateau = merge_dataclass(PlateauThresholds, defaults.plateau, data.get("plateau", {}))
        dt = merge_dataclass(DtThresholds, defaults.dt, data.get("dt", {}))
        integrity = merge_dataclass(
            IntegrityThresholds, defaults.integrity, data.get("integrity", {})
        )
        return MetricsThresholds(
            milestones=milestones,
            safety=safety,
            stability=stability,
            altitude=altitude,
            plateau=plateau,
            dt=dt,
            integrity=integrity,
        )

    # -------------------------
    # Lifecycle
    # -------------------------

    def set_t0(self, t0_s: float) -> None:
        """Start metrics at sim-time t0_s (post PASS gate)."""
        self.t0_s = float(t0_s)
        self._last_sim_time_s = None
        self._last_valid_time_s = None
        self._invalid_streak_s = 0.0
        self.sim_time_end_s = None

    def update_geometry(
        self,
        *,
        sim_time_s: float,
        behind_tip_m: float,
        cross_tip_m: float,
        below_tip_m: float,
        range_tip_m: float,
        behind_center_m: float,
        cross_center_m: float,
        below_center_m: float,
        range_center_m: float,
        controller_state: str,
        follower_alt_msl: Optional[float] = None,
        closure_rate_mps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update metrics with a single controller-loop sample.

        Returns:
            Dict with optional keys:
              - alerts: list[str] of log-ready messages (milestones, safety warnings)
        """

        alerts: List[str] = []

        # Ignore everything until PASS gate timebase start.
        if self.t0_s is None:
            return {"alerts": alerts}

        sim_time_s = float(sim_time_s)
        self.sim_time_end_s = sim_time_s

        # dt from sim-time.
        dt = None
        if self._last_sim_time_s is not None:
            dt = sim_time_s - self._last_sim_time_s
        self._last_sim_time_s = sim_time_s

        self._sample_count += 1
        self._sample_count_total += 1
        # P8.4 v9.8: Only include positive dt (filter non-monotonic timestamps)
        if dt is not None and _is_finite(dt) and dt > 0:
            self._dt_values.append(float(dt))

        # Track tip/center geometry validity (for integrity metrics)
        tip_sane = _geom_is_sane(
            behind_m=behind_tip_m,
            cross_m=cross_tip_m,
            below_m=below_tip_m,
            range_m=range_tip_m,
            margin_m=self.th.safety.geom_sanity_margin_m,
        )
        center_sane = _geom_is_sane(
            behind_m=behind_center_m,
            cross_m=cross_center_m,
            below_m=below_center_m,
            range_m=range_center_m,
            margin_m=self.th.safety.geom_sanity_margin_m,
        )
        if tip_sane:
            self._sample_count_valid_tip += 1
        if center_sane:
            self._sample_count_valid_center += 1

        # Basic validity checks.
        required = [
            behind_tip_m,
            cross_tip_m,
            below_tip_m,
            range_tip_m,
            behind_center_m,
            cross_center_m,
            below_center_m,
            range_center_m,
        ]
        valid = all(_is_finite(v) for v in required)

        # Compute integration dt (clamped) only when dt>0.
        dt_int: Optional[float] = None
        if dt is not None and _is_finite(dt) and dt > 0:
            dt_int = _clamp(float(dt), self.th.dt.dt_min_s, self.th.dt.dt_max_s)

        if not valid:
            self._invalid_sample_count += 1
            if dt_int is not None:
                self._invalid_streak_s += dt_int
            # Hard fail if invalid streak persists.
            if (
                not self.hard_fail
                and self._invalid_streak_s >= self.th.safety.nan_persist_hard_fail_s
            ):
                self._register_hard_fail(
                    code=SafetyCode.S3_NAN_PERSIST,
                    reason=f"Invalid samples persisted for {self._invalid_streak_s:.2f}s",
                    sim_time_s=sim_time_s,
                    duration_s=self._invalid_streak_s,
                )
                alerts.append(
                    f"[SAFETY_{SafetyCode.S3_NAN_PERSIST.value}] t={_now_rel(sim_time_s, self.t0_s):.3f}s duration={self._invalid_streak_s:.3f}s"
                )
            return {"alerts": alerts}

        # If valid, reset invalid streak.
        self._valid_sample_count += 1
        self._invalid_streak_s = 0.0
        self._last_valid_time_s = sim_time_s

        # Compute max range jump (telemetry sanity).
        prev_range_center_m = self._last_range_center_m  # save for closure rate in _safety_checks
        if self._last_range_center_m is not None and _is_finite(self._last_range_center_m):
            jump = abs(float(range_center_m) - float(self._last_range_center_m))
            if self._max_range_jump_m is None or jump > self._max_range_jump_m:
                self._max_range_jump_m = jump
        self._last_range_center_m = float(range_center_m)

        # Always update min/max tracking (valid sample).
        # P6.2 v9.6: Added controller_state for context tracking
        self._update_range_tracking(
            sim_time_s=sim_time_s,
            behind_tip_m=float(behind_tip_m),
            cross_tip_m=float(cross_tip_m),
            below_tip_m=float(below_tip_m),
            range_tip_m=float(range_tip_m),
            behind_center_m=float(behind_center_m),
            range_center_m=float(range_center_m),
            controller_state=str(controller_state),
        )

        # If dt is non-positive, do not integrate timers or stability buffers.
        if dt_int is None:
            return {"alerts": alerts}

        # Safety checks (leader-center frame).
        alerts.extend(
            self._safety_checks(
                sim_time_s=sim_time_s,
                dt_int=dt_int,
                behind_center_m=float(behind_center_m),
                cross_center_m=float(cross_center_m),
                below_center_m=float(below_center_m),
                range_center_m=float(range_center_m),
                prev_range_center_m=prev_range_center_m,
                controller_state=str(controller_state),
            )
        )

        # If hard fail already occurred, we still track dt/range stats but skip milestones/stability.
        if self.hard_fail:
            return {"alerts": alerts}

        # Milestones (M1/M2 use tip frame; M3a/M3b use boom-tip-frame deltas from center).
        alerts.extend(
            self._milestone_checks(
                sim_time_s=sim_time_s,
                dt_int=dt_int,
                behind_tip_m=float(behind_tip_m),
                cross_tip_m=float(cross_tip_m),
                below_tip_m=float(below_tip_m),
                behind_center_m=float(behind_center_m),
                cross_center_m=float(cross_center_m),
                below_center_m=float(below_center_m),
                controller_state=str(controller_state),
            )
        )

        # Stability (phase-scoped rolling window).
        self._update_stability(
            sim_time_s=sim_time_s,
            controller_state=str(controller_state),
            behind_tip_m=float(behind_tip_m),
            cross_tip_m=float(cross_tip_m),
            below_tip_m=float(below_tip_m),
        )

        # Altitude tracking (relative by default).
        self._update_altitude(
            sim_time_s=sim_time_s,
            dt_int=dt_int,
            controller_state=str(controller_state),
            below_tip_m=float(below_tip_m),
            follower_alt_msl=None if follower_alt_msl is None else float(follower_alt_msl),
        )

        # Plateau tracking (optional) using closure rate.
        self._update_plateau(
            sim_time_s=sim_time_s,
            dt_int=dt_int,
            controller_state=str(controller_state),
            behind_tip_m=float(behind_tip_m),
            closure_rate_mps=None if closure_rate_mps is None else float(closure_rate_mps),
        )

        return {"alerts": alerts}

    def on_state_transition(
        self, from_state: str, to_state: str, reason: str = "", sim_time_s: Optional[float] = None
    ) -> None:
        """Record state transition with sim-time timestamp.

        Rec 5: Builds a sim-time indexed timeline of state transitions
        for cross-speedup comparison. Timestamps are relative to t0.
        """
        if self.t0_s is not None and sim_time_s is not None:
            t_rel = _now_rel(sim_time_s, self.t0_s)
        elif self._last_sim_time_s is not None and self.t0_s is not None:
            t_rel = _now_rel(self._last_sim_time_s, self.t0_s)
        else:
            t_rel = None
        self._state_timeline.append(
            {
                "t_s": round(t_rel, 3) if t_rel is not None else None,
                "from": str(from_state),
                "to": str(to_state),
                "reason": str(reason),
            }
        )

    def set_speed_arbitration_stats(self, stats: Dict[str, Any]) -> None:
        """P2 v9.4: Store speed arbitration stats from controller."""
        self._speed_arb_stats = stats

    def set_dc_phase_stats(self, stats: Dict[str, Any]) -> None:
        """Option A+E: Store DOCKING_CREEP sub-phase and TECS stats."""
        self._dc_phase_stats = stats

    # -------------------------
    # Tranche D: Public accessors for per-tick M3 state
    # -------------------------

    def get_m3_per_tick_state(self) -> Dict[str, Any]:
        """Return per-tick M3a/M3b dwell state for CSV output (D4)."""
        return {
            "delta_behind": self._last_delta_behind,
            "delta_below": self._last_delta_below,
            "delta_cross": self._last_delta_cross,
            "m3a_in_envelope": self.m3a._current_dwell_s > 0.0,
            "m3a_dwell_raw_s": self._m3a_dwell_raw_s,
            "m3a_dwell_grace_s": self._m3a_dwell_grace_s,
            "m3b_in_envelope": self.m3b._current_dwell_s > 0.0,
            "m3b_dwell_raw_s": self._m3b_dwell_raw_s,
            "m3b_dwell_grace_s": self._m3b_dwell_grace_s,
            # Step 1a: M3a_coarse_delta tracking
            "m3a_cd_in_envelope": self.m3a_coarse_delta._current_dwell_s > 0.0,
            "m3a_cd_dwell_raw_s": self._m3a_cd_dwell_raw_s,
        }

    def _finalize_open_segments(self) -> None:
        """O-FIX: Close any still-open dwell segments at run end."""
        t0 = self.t0_s if self.t0_s is not None else 0.0
        t_end = (float(self.sim_time_end_s) if self.sim_time_end_s is not None else t0) - t0
        for prefix in ("m3a", "m3b", "m3a_cd"):
            seg_start = getattr(self, f"_{prefix}_seg_start")
            if seg_start is not None:
                segments = getattr(self, f"_{prefix}_segments")
                segments.append(
                    DwellSegment(
                        t_start=seg_start,
                        t_end=t_end,
                        duration_s=t_end - seg_start,
                        max_abs_delta_behind=getattr(self, f"_{prefix}_seg_max_behind"),
                        max_abs_delta_below=getattr(self, f"_{prefix}_seg_max_below"),
                        max_abs_delta_cross=getattr(self, f"_{prefix}_seg_max_cross"),
                        state_at_entry=getattr(self, f"_{prefix}_seg_state"),
                    )
                )
                setattr(self, f"_{prefix}_seg_start", None)

    def finalize(self) -> OutcomeClass:
        """Compute final outcome classification."""
        if self.t0_s is None:
            # Metrics never started; treat as FAIL.
            self.outcome = OutcomeClass.FAIL
            return self.outcome

        # O-FIX: Close any open dwell segments before outcome computation
        self._finalize_open_segments()

        # If hard fail, map to outcome.
        if self.hard_fail:
            self.outcome = self._map_hard_fail_to_outcome()
            return self.outcome

        # Success precedence.
        if self.m3a.achieved:
            self.outcome = OutcomeClass.SUCCESS_CONTACT
            return self.outcome
        if self.m2.achieved:
            self.outcome = OutcomeClass.SUCCESS_PRECONTACT_ONLY
            return self.outcome

        # Non-success classification.
        # Plateau check: if M2 missed deadline and plateau condition detected.
        if self._is_plateau():
            self.outcome = OutcomeClass.PLATEAU
            return self.outcome

        # No milestone achieved, no plateau, no hard-fail -> INCOMPLETE
        self.outcome = OutcomeClass.INCOMPLETE
        return self.outcome

    def to_dict(self) -> Dict[str, Any]:
        """Export summary dict matching run_summary.schema.json."""
        if self.t0_s is None:
            t0 = 0.0
        else:
            t0 = float(self.t0_s)

        sim_end = float(self.sim_time_end_s) if self.sim_time_end_s is not None else t0

        # P3 v9.4: Enhanced dt stats
        dt_stats: Optional[Dict[str, Any]] = None
        if self._dt_values:
            dts = [d for d in self._dt_values if _is_finite(d)]
            if dts:
                dts_sorted = sorted(dts)
                n = len(dts_sorted)
                dt_mean = sum(dts_sorted) / n

                # Percentiles
                p50 = dts_sorted[int(0.50 * (n - 1))] if n > 0 else 0.0
                p95 = dts_sorted[int(0.95 * (n - 1))] if n > 0 else 0.0
                p99 = dts_sorted[int(0.99 * (n - 1))] if n > 0 else 0.0

                # Spike count (dt > 0.2s indicates potential stutter)
                spike_count = sum(1 for d in dts if d > 0.2)

                # P8.4 v9.8: Clamp counts - only count positive dt values
                clamp_low_count = sum(
                    1
                    for d in self._dt_values
                    if _is_finite(d) and d > 0 and d < self.th.dt.dt_min_s
                )
                clamp_high_count = sum(
                    1
                    for d in self._dt_values
                    if _is_finite(d) and d > 0 and d > self.th.dt.dt_max_s
                )

                dt_stats = {
                    "sample_count": int(self._sample_count),
                    "valid_sample_count": int(self._valid_sample_count),
                    "invalid_sample_count": int(self._invalid_sample_count),
                    "dt_min_s": min(dts_sorted),
                    "dt_max_s": max(dts_sorted),
                    "mean_s": dt_mean,
                    "p50_s": float(p50),
                    "p95_s": float(p95),
                    "p99_s": float(p99),
                    "spike_count": spike_count,  # dt > 0.2s
                    "effective_hz_mean": 1.0 / dt_mean if dt_mean > 0 else 0.0,
                    "clamp_low_count": clamp_low_count,  # dt < DT_MIN
                    "clamp_high_count": clamp_high_count,  # dt > DT_MAX (before clamp)
                    # Backwards compatibility
                    "dt_mean_s": dt_mean,
                    "dt_p95_s": float(p95),
                }

        thresholds_dict = {
            "milestones": asdict(self.th.milestones),
            "safety": {
                **asdict(self.th.safety),
                "hard_fail_states": list(self.th.safety.hard_fail_states),
                "safety_eval_states": list(self.th.safety.safety_eval_states),
            },
            "stability": {
                **asdict(self.th.stability),
                "precontact_states": list(self.th.stability.precontact_states),
                "contact_states": list(self.th.stability.contact_states),
            },
            "altitude": asdict(self.th.altitude),
            "plateau": {
                **asdict(self.th.plateau),
                "plateau_states": list(self.th.plateau.plateau_states),
            },
            "dt": asdict(self.th.dt),
        }

        # Ensure outcome is computed.
        if self.outcome is None:
            self.finalize()

        # P6.4 v9.6: Top-level speedup invariance summary for easy cross-run comparison
        speedup_invariance: Optional[Dict[str, Any]] = None
        if dt_stats:
            eff_hz = dt_stats.get("effective_hz_mean", 0.0)
            dt_mean = dt_stats.get("mean_s", 0.0)
            dt_p95 = dt_stats.get("p95_s", 0.0)
            spike_count = dt_stats.get("spike_count", 0)
            sample_count = dt_stats.get("sample_count", 1)
            spike_pct = 100.0 * spike_count / max(1, sample_count)
            speedup_invariance = {
                "effective_hz_mean": round(eff_hz, 2),
                "dt_mean_s": round(dt_mean, 4),
                "dt_p95_s": round(dt_p95, 4),
                "spike_count": spike_count,
                "spike_pct": round(spike_pct, 2),
                "clamp_low_count": dt_stats.get("clamp_low_count", 0),
                "clamp_high_count": dt_stats.get("clamp_high_count", 0),
                "sample_count": sample_count,
            }

        return {
            "spec_version": self.SPEC_VERSION,
            # P7.5 v9.7: Git metadata for traceability
            "git_commit": self._git_metadata.get("git_commit", "unknown"),
            "working_tree_clean": self._git_metadata.get("working_tree_clean"),
            "t0_s": t0,
            "sim_time_end_s": sim_end,
            # P6.4 v9.6: Speedup invariance metrics at top level for visibility
            "speedup_invariance": speedup_invariance,
            "dt_stats": dt_stats,
            "thresholds": thresholds_dict,
            "milestones": {
                "m1_staging": self.m1.to_public_dict(),
                "m2_precontact": self.m2.to_public_dict(),
                "m3a_coarse_contact": self.m3a.to_public_dict(),
                # Step 1a: Correctly-named fine tier (same data as m3a_coarse_contact)
                "m3a_fine_contact": {
                    **self.m3a.to_public_dict(),
                    "thresholds_used": {
                        "delta_behind_tol": self.th.milestones.m3a_delta_behind_tol,
                        "delta_below_tol": self.th.milestones.m3a_delta_below_tol,
                        "delta_cross_tol": self.th.milestones.m3a_delta_cross_tol,
                        "dwell_s": self.th.milestones.m3a_dwell_s,
                    },
                },
                # Step 1a: Actual coarse delta tier (wider thresholds)
                "m3a_coarse_delta_contact": {
                    **self.m3a_coarse_delta.to_public_dict(),
                    "thresholds_used": {
                        "delta_behind_tol": self.th.milestones.m3a_coarse_delta_behind_tol,
                        "delta_below_tol": self.th.milestones.m3a_coarse_delta_below_tol,
                        "delta_cross_tol": self.th.milestones.m3a_coarse_delta_cross_tol,
                        "dwell_s": self.th.milestones.m3a_coarse_dwell_s,
                    },
                },
                "m3b_tight_contact": {
                    **self.m3b.to_public_dict(),
                    "thresholds_used": {
                        "delta_behind_tol": self.th.milestones.m3b_delta_behind_tol,
                        "delta_below_tol": self.th.milestones.m3b_delta_below_tol,
                        "delta_cross_tol": self.th.milestones.m3b_delta_cross_tol,
                        "dwell_s": self.th.milestones.m3b_dwell_s,
                    },
                },
            },
            # O-FIX: Dwell segment proof instrumentation
            "dwell_segments": {
                "m3a_fine": [s.to_dict() for s in self._m3a_segments],
                "m3a_coarse_delta": [s.to_dict() for s in self._m3a_cd_segments],
                "m3b_tight": [s.to_dict() for s in self._m3b_segments],
                "m3a_fine_count": len(self._m3a_segments),
                "m3a_coarse_delta_count": len(self._m3a_cd_segments),
                "m3b_tight_count": len(self._m3b_segments),
                "m3a_fine_total_dwell_s": round(sum(s.duration_s for s in self._m3a_segments), 3),
                "m3a_coarse_delta_total_dwell_s": round(
                    sum(s.duration_s for s in self._m3a_cd_segments), 3
                ),
                "m3b_tight_total_dwell_s": round(sum(s.duration_s for s in self._m3b_segments), 3),
            },
            "safety": {
                "hard_fail": bool(self.hard_fail),
                "hard_fail_reason": str(self.hard_fail_reason),
                "violations": [e.to_public_dict() for e in self.safety_events],
            },
            "stability": {
                "precontact": dict(self._stab_pre.latest),
                "contact": dict(self._stab_con.latest),
            },
            "altitude": {
                "mode": self.th.altitude.mode,
                "precontact_violation_time_s": float(self.precontact_violation_time_s),
                "contact_violation_time_s": float(self.contact_violation_time_s),
            },
            "range_tracking": {
                "min_range_tip_m": self.min_range_tip_m,
                "min_range_tip_time_s": self.min_range_tip_time_s,
                "min_range_center_m": self.min_range_center_m,
                "min_range_center_time_s": self.min_range_center_time_s,
                "min_behind_tip_m": self.min_behind_tip_m,
                "min_behind_tip_time_s": self.min_behind_tip_time_s,
                # P6.2 v9.6: Context at min_behind_tip for diagnosis
                "state_at_min_behind_tip": self.state_at_min_behind_tip,
                "below_m_at_min_behind_tip": self.below_m_at_min_behind_tip,
                "cross_m_at_min_behind_tip": self.cross_m_at_min_behind_tip,
                "min_behind_center_m": self.min_behind_center_m,
                "min_behind_center_time_s": self.min_behind_center_time_s,
                "max_abs_cross_tip_m": self.max_abs_cross_tip_m,
                "max_abs_below_tip_m": self.max_abs_below_tip_m,
                "max_range_jump_m": self._max_range_jump_m,
            },
            "geometry_sanity": {
                "center_insane_count": self._center_insane_count,
                "center_insane_first_s": self._center_insane_first_s,
                "safety_skipped_far_count": self._safety_skipped_far_count,
                "s1_skipped_state_count": self._s1_skipped_state_count,
                "safety_evaluated_count": self._safety_evaluated_count,
                "s1_evaluated_count": self._s1_evaluated_count,
                "s2_evaluated_count": self._s2_evaluated_count,
            },
            "integrity": self._compute_integrity_block(),
            "plateau": {
                "detected": self._plateau_detected,
                "first_detected_s": self._plateau_first_detected_s,
                "accum_s": self._plateau_accum_s,
                "window_s": self.th.plateau.plateau_window_s,
                "behind_min_m": self.th.plateau.behind_min_m,
                "closure_rate_tol_mps": self.th.plateau.closure_rate_abs_tol_mps,
                "sample_count": len(self._closure_rate_hist),
            },
            # P2 v9.4: Speed arbitration stats
            "speed_arbitration": (
                self._speed_arb_stats
                if self._speed_arb_stats
                else {
                    "total_ticks": 0,
                    "pct_dominant_profile": 0.0,
                    "pct_dominant_brake": 0.0,
                    "pct_dominant_climb": 0.0,
                    "pct_dominant_safety": 0.0,
                }
            ),
            # Rec 5: Sim-time indexed state transition timeline
            "state_timeline": list(self._state_timeline),
            # Option A+E: DOCKING_CREEP sub-phase + TECS stats
            "dc_phase_stats": self._dc_phase_stats if self._dc_phase_stats else {},
            # Tranche D: Boom-tip-frame milestone context
            "boom_tip_nominal": {
                "behind_m": self._boom_tip_behind,
                "below_m": self._boom_tip_below,
                "cross_m": self._boom_tip_cross,
            },
            "outcome": self.outcome.value if self.outcome else OutcomeClass.INCOMPLETE.value,
        }

    # -------------------------
    # Internal: integrity
    # -------------------------

    def _compute_integrity_block(self) -> Dict[str, Any]:
        """Compute integrity metrics with pass/fail checks."""
        th = self.th.integrity

        # Compute percentages
        pct_valid_tip = (
            100.0 * self._sample_count_valid_tip / self._sample_count_total
            if self._sample_count_total > 0
            else None
        )
        pct_valid_center = (
            100.0 * self._sample_count_valid_center / self._sample_count_total
            if self._sample_count_total > 0
            else None
        )
        # Safety evaluated percentage: denominator is valid_center samples
        pct_safety_evaluated = (
            100.0 * self._safety_evaluated_count / self._sample_count_valid_center
            if self._sample_count_valid_center > 0
            else None
        )

        # Check integrity against thresholds (structured for machine parsing)
        failed_checks: List[Dict[str, Any]] = []

        if self._sample_count_total < th.min_sample_count:
            failed_checks.append(
                {
                    "check": "min_sample_count",
                    "value": self._sample_count_total,
                    "threshold": th.min_sample_count,
                    "value_type": "count",
                    "unit": "samples",
                }
            )

        if pct_valid_tip is not None and pct_valid_tip < th.min_percent_valid_tip:
            failed_checks.append(
                {
                    "check": "min_percent_valid_tip",
                    "value": pct_valid_tip,
                    "threshold": th.min_percent_valid_tip,
                    "value_type": "percent",
                    "unit": "%",
                }
            )

        if pct_valid_center is not None and pct_valid_center < th.min_percent_valid_center:
            failed_checks.append(
                {
                    "check": "min_percent_valid_center",
                    "value": pct_valid_center,
                    "threshold": th.min_percent_valid_center,
                    "value_type": "percent",
                    "unit": "%",
                }
            )

        if (
            pct_safety_evaluated is not None
            and pct_safety_evaluated < th.min_percent_safety_evaluated
        ):
            failed_checks.append(
                {
                    "check": "min_percent_safety_evaluated",
                    "value": pct_safety_evaluated,
                    "threshold": th.min_percent_safety_evaluated,
                    "value_type": "percent",
                    "unit": "%",
                }
            )

        integrity_ok = len(failed_checks) == 0

        return {
            "spec_version": "1.0",  # For standalone extraction
            # Counts
            "sample_count_total": self._sample_count_total,
            "sample_count_valid_tip": self._sample_count_valid_tip,
            "sample_count_valid_center": self._sample_count_valid_center,
            # Percentages
            "percent_valid_tip": pct_valid_tip,
            "percent_valid_center": pct_valid_center,
            # Safety eval with explicit denominator
            "safety_evaluated_count": self._safety_evaluated_count,
            "safety_eval_denominator": "valid_center",
            "safety_eval_denominator_count": self._sample_count_valid_center,
            "percent_safety_evaluated": pct_safety_evaluated,
            # Pass/fail
            "integrity_ok": integrity_ok,
            "failed_checks": failed_checks,
            # Thresholds used (for reproducibility)
            "thresholds": {
                "min_sample_count": th.min_sample_count,
                "min_percent_valid_tip": th.min_percent_valid_tip,
                "min_percent_valid_center": th.min_percent_valid_center,
                "min_percent_safety_evaluated": th.min_percent_safety_evaluated,
            },
        }

    # -------------------------
    # Internal: tracking
    # -------------------------

    def _update_range_tracking(
        self,
        *,
        sim_time_s: float,
        behind_tip_m: float,
        cross_tip_m: float,
        below_tip_m: float,
        range_tip_m: float,
        behind_center_m: float,
        range_center_m: float,
        controller_state: str = "",
    ) -> None:
        # mins - record timestamp when new minimum found
        if self.min_range_tip_m is None or range_tip_m < self.min_range_tip_m:
            self.min_range_tip_m = range_tip_m
            self.min_range_tip_time_s = _now_rel(sim_time_s, self.t0_s)
            self._min_range_tip_state = controller_state  # Track state at min range
        if self.min_range_center_m is None or range_center_m < self.min_range_center_m:
            self.min_range_center_m = range_center_m
            self.min_range_center_time_s = _now_rel(sim_time_s, self.t0_s)

        # P8.4 v9.8: Only record min_behind in safety_eval_states
        # This filters out spurious minima recorded during approach phases (e.g., STERN_CONVERSION)
        # where the follower may be geometrically "close" but not in a formation-relevant state.
        in_safety_state = controller_state in set(self.th.safety.safety_eval_states)
        if in_safety_state:
            if self.min_behind_tip_m is None or behind_tip_m < self.min_behind_tip_m:
                self.min_behind_tip_m = behind_tip_m
                self.min_behind_tip_time_s = _now_rel(sim_time_s, self.t0_s)
                # P6.2 v9.6: Capture context at min_behind_tip
                self.state_at_min_behind_tip = controller_state
                self.below_m_at_min_behind_tip = below_tip_m
                self.cross_m_at_min_behind_tip = cross_tip_m
            if self.min_behind_center_m is None or behind_center_m < self.min_behind_center_m:
                self.min_behind_center_m = behind_center_m
                self.min_behind_center_time_s = _now_rel(sim_time_s, self.t0_s)

        # max abs
        abs_cross = abs(cross_tip_m)
        abs_below = abs(below_tip_m)
        if self.max_abs_cross_tip_m is None or abs_cross > self.max_abs_cross_tip_m:
            self.max_abs_cross_tip_m = abs_cross
        if self.max_abs_below_tip_m is None or abs_below > self.max_abs_below_tip_m:
            self.max_abs_below_tip_m = abs_below

    def _register_hard_fail(
        self,
        *,
        code: SafetyCode,
        reason: str,
        sim_time_s: float,
        duration_s: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.hard_fail = True
        self.hard_fail_reason = reason
        self.safety_events.append(
            SafetyEvent(
                code=code,
                severity="HARD_FAIL",
                first_time_s=_now_rel(
                    sim_time_s, self.t0_s if self.t0_s is not None else sim_time_s
                ),
                duration_s=duration_s,
                details=details,
            )
        )

    def _register_warn(
        self,
        *,
        code: SafetyCode,
        sim_time_s: float,
        duration_s: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.safety_events.append(
            SafetyEvent(
                code=code,
                severity="WARN",
                first_time_s=_now_rel(
                    sim_time_s, self.t0_s if self.t0_s is not None else sim_time_s
                ),
                duration_s=duration_s,
                details=details,
            )
        )

    # -------------------------
    # Internal: safety
    # -------------------------

    def _safety_checks(
        self,
        *,
        sim_time_s: float,
        dt_int: float,
        behind_center_m: float,
        cross_center_m: float,
        below_center_m: float,
        range_center_m: float,
        prev_range_center_m: Optional[float],
        controller_state: str,
    ) -> List[str]:
        alerts: List[str] = []

        # S4: controller fail state
        if (not self.hard_fail) and controller_state in set(self.th.safety.hard_fail_states):
            self._register_hard_fail(
                code=SafetyCode.S3_STATE_FAIL,
                reason=f"Controller entered hard-fail state: {controller_state}",
                sim_time_s=sim_time_s,
                details={"state": controller_state},
            )
            alerts.append(
                f"[SAFETY_{SafetyCode.S3_STATE_FAIL.value}] t={_now_rel(sim_time_s, self.t0_s):.3f}s state={controller_state}"
            )
            return alerts

        # Gate 1: Geometry sanity - skip if center geometry is invalid (applies to both S1/S2)
        if not _geom_is_sane(
            behind_m=behind_center_m,
            cross_m=cross_center_m,
            below_m=below_center_m,
            range_m=range_center_m,
            margin_m=self.th.safety.geom_sanity_margin_m,
        ):
            t_rel = _now_rel(sim_time_s, self.t0_s)
            self._center_insane_count += 1
            if self._center_insane_first_s is None:
                self._center_insane_first_s = t_rel
            return alerts

        # S1: overtake - gated by state AND proximity
        # (overtake is formation-contextual, so state gating makes sense)
        skip_s1_state = controller_state not in self.th.safety.safety_eval_states
        skip_s1_range = range_center_m > self.th.safety.overtake_eval_max_range_m
        skip_s1 = skip_s1_state or skip_s1_range

        if skip_s1_state:
            self._s1_skipped_state_count += 1

        # S2: collision - gated by proximity ONLY (not state)
        # (collision is ALWAYS safety-relevant regardless of controller state)
        skip_s2 = range_center_m > self.th.safety.collision_eval_max_range_m

        # Track far skips (only when both S1 and S2 skipped due to range)
        if skip_s1_range and skip_s2:
            self._safety_skipped_far_count += 1

        # S1: overtake (persist) - only if NOT skipped by state or range
        if not skip_s1:
            self._s1_evaluated_count += 1
            if behind_center_m < self.th.safety.overtake_behind_center_m:
                self._overtake_accum_s += dt_int
                if (
                    not self.hard_fail
                ) and self._overtake_accum_s >= self.th.safety.overtake_persist_s:
                    self._register_hard_fail(
                        code=SafetyCode.S1_OVERTAKE,
                        reason=f"Overtake persisted {self._overtake_accum_s:.2f}s (behind_center_m={behind_center_m:.2f})",
                        sim_time_s=sim_time_s,
                        duration_s=self._overtake_accum_s,
                        details={"behind_center_m": behind_center_m},
                    )
                    alerts.append(
                        f"[SAFETY_{SafetyCode.S1_OVERTAKE.value}] t={_now_rel(sim_time_s, self.t0_s):.3f}s duration={self._overtake_accum_s:.3f}s behind_center_m={behind_center_m:.3f}"
                    )
            else:
                self._overtake_accum_s = 0.0

        # S2: collision — two-tier (proximity + closure rate)
        if not skip_s2:
            self._s2_evaluated_count += 1

            # Compute range closure rate: positive = closing (range decreasing)
            # Matches codebase convention (geometry.py:239 "positive = closing")
            range_closure_rate = None
            if prev_range_center_m is not None and dt_int is not None and dt_int > 0:
                range_closure_rate = (prev_range_center_m - range_center_m) / dt_int

            # --- Tier 1: Hard floor (unconditional) ---
            if range_center_m < self.th.safety.collision_hard_floor_m:
                self._collision_hard_floor_accum_s += dt_int
                if (
                    (not self.hard_fail)
                    and self._collision_hard_floor_accum_s
                    >= self.th.safety.collision_hard_floor_persist_s
                ):
                    self._register_hard_fail(
                        code=SafetyCode.S2_COLLISION,
                        reason=(
                            f"Hard-floor collision: range {range_center_m:.2f}m < "
                            f"{self.th.safety.collision_hard_floor_m:.1f}m for "
                            f"{self._collision_hard_floor_accum_s:.2f}s"
                        ),
                        sim_time_s=sim_time_s,
                        duration_s=self._collision_hard_floor_accum_s,
                        details={
                            "range_center_m": range_center_m,
                            "tier": "hard_floor",
                        },
                    )
                    alerts.append(
                        f"[SAFETY_{SafetyCode.S2_COLLISION.value}] t={_now_rel(sim_time_s, self.t0_s):.3f}s "
                        f"HARD_FLOOR duration={self._collision_hard_floor_accum_s:.3f}s "
                        f"range_center_m={range_center_m:.3f}"
                    )
            else:
                self._collision_hard_floor_accum_s = 0.0

            # --- Tier 2: Proximity + uncontrolled closure ---
            # Triggers when inside collision_range_center_m AND closing faster than threshold
            closing_fast = (
                range_closure_rate is not None
                and range_closure_rate > self.th.safety.collision_closure_rate_mps
            )
            if range_center_m < self.th.safety.collision_range_center_m and closing_fast:
                self._collision_accum_s += dt_int
                if (
                    not self.hard_fail
                ) and self._collision_accum_s >= self.th.safety.collision_persist_s:
                    self._register_hard_fail(
                        code=SafetyCode.S2_COLLISION,
                        reason=(
                            f"Uncontrolled closure: range {range_center_m:.2f}m, "
                            f"closure_rate={range_closure_rate:.2f} m/s for "
                            f"{self._collision_accum_s:.2f}s"
                        ),
                        sim_time_s=sim_time_s,
                        duration_s=self._collision_accum_s,
                        details={
                            "range_center_m": range_center_m,
                            "range_closure_rate_mps": range_closure_rate,
                            "tier": "closure_gated",
                        },
                    )
                    alerts.append(
                        f"[SAFETY_{SafetyCode.S2_COLLISION.value}] t={_now_rel(sim_time_s, self.t0_s):.3f}s "
                        f"CLOSURE_GATED duration={self._collision_accum_s:.3f}s "
                        f"range_center_m={range_center_m:.3f} closure={range_closure_rate:.2f}m/s"
                    )
            else:
                self._collision_accum_s = 0.0

            if (
                not self._near_miss_seen
            ) and range_center_m < self.th.safety.near_miss_range_center_m:
                self._near_miss_seen = True
                self._register_warn(
                    code=SafetyCode.S2_NEAR_MISS,
                    sim_time_s=sim_time_s,
                    details={"range_center_m": range_center_m},
                )
                alerts.append(
                    f"[SAFETY_{SafetyCode.S2_NEAR_MISS.value}] t={_now_rel(sim_time_s, self.t0_s):.3f}s range_center_m={range_center_m:.3f}"
                )

        # Count as evaluated if either S1 or S2 actually ran
        if (not skip_s1) or (not skip_s2):
            self._safety_evaluated_count += 1

        return alerts

    # -------------------------
    # Internal: milestones
    # -------------------------

    def _milestone_checks(
        self,
        *,
        sim_time_s: float,
        dt_int: float,
        behind_tip_m: float,
        cross_tip_m: float,
        below_tip_m: float,
        behind_center_m: float = 0.0,
        cross_center_m: float = 0.0,
        below_center_m: float = 0.0,
        controller_state: str,
    ) -> List[str]:
        alerts: List[str] = []
        t_rel = _now_rel(sim_time_s, self.t0_s)

        # M1 staging corridor (uses tip-frame, unchanged — large range, offset irrelevant)
        in_m1 = (
            self.th.milestones.m1_behind_min <= behind_tip_m <= self.th.milestones.m1_behind_max
            and abs(cross_tip_m) <= self.th.milestones.m1_cross_tol
            and abs(below_tip_m) <= self.th.milestones.m1_below_tol
        )
        if in_m1:
            if self.m1.first_entry_s is None:
                self.m1.first_entry_s = t_rel
                self.m1.entries += 1
                alerts.append(f"[MILESTONE_M1] t={t_rel:.3f}s")
            # Phase 0 contract: M1 is entry-only (no dwell); mark achieved on first entry.
            if not self.m1.achieved:
                self.m1.achieved = True
                self.m1.first_dwell_complete_s = t_rel
            self.m1.best_dwell_s = max(self.m1.best_dwell_s, 0.0)

        # M2 precontact hold (uses tip-frame, unchanged)
        m2_lo = self.th.milestones.m2_behind_target - self.th.milestones.m2_behind_tol
        m2_hi = self.th.milestones.m2_behind_target + self.th.milestones.m2_behind_tol
        in_m2 = (
            m2_lo <= behind_tip_m <= m2_hi
            and abs(cross_tip_m) <= self.th.milestones.m2_cross_tol
            and (self.th.milestones.m2_below_target - self.th.milestones.m2_below_tol)
            <= below_tip_m
            <= (self.th.milestones.m2_below_target + self.th.milestones.m2_below_tol)
        )
        alerts.extend(
            self._update_milestone_dwell(
                ms=self.m2,
                in_env=in_m2,
                dt_int=dt_int,
                t_rel=t_rel,
                dwell_req=self.th.milestones.m2_dwell_s,
                tag="M2",
            )
        )

        # Tranche D: Boom-tip-frame deltas.
        # behind_tip_m, below_tip_m, cross_tip_m are already measured FROM the
        # boom tip (via fused geometry). Using these directly avoids the ~9m
        # raw-GPS latency error seen when computing deltas from un-fused
        # center-CG geometry (behind_center_m - boom_tip_nominal).
        delta_behind = behind_tip_m
        delta_below = below_tip_m
        delta_cross = cross_tip_m

        # Cache for CSV output (D4)
        self._last_delta_behind = delta_behind
        self._last_delta_below = delta_below
        self._last_delta_cross = delta_cross

        # O-FIX: Split M3 gate into entry (broad) vs dwell (strict)
        # allow_m3_entry: detects envelope entries even if state machine lags (M2 latch OR contact)
        # allow_m3_dwell: only accumulates dwell in contact states (DOCKING_CREEP, HOLD_CONTACT)
        allow_m3_entry = self.m2.achieved or (
            controller_state in set(self.th.stability.contact_states)
        )
        allow_m3_dwell = controller_state in set(self.th.stability.contact_states)
        if allow_m3_entry:
            # Tranche D: boom-tip-frame envelope (tolerances around zero)
            in_m3a = (
                abs(delta_behind) <= self.th.milestones.m3a_delta_behind_tol
                and abs(delta_below) <= self.th.milestones.m3a_delta_below_tol
                and abs(delta_cross) <= self.th.milestones.m3a_delta_cross_tol
            )
            alerts.extend(
                self._update_milestone_dwell(
                    ms=self.m3a,
                    in_env=in_m3a,
                    dt_int=dt_int,
                    t_rel=t_rel,
                    dwell_req=self.th.milestones.m3a_dwell_s,
                    tag="M3A",
                    allow_dwell=allow_m3_dwell,
                )
            )
            self._update_dwell_segment(
                "m3a",
                in_m3a and allow_m3_dwell,
                t_rel,
                delta_behind,
                delta_below,
                delta_cross,
                controller_state,
            )

            # D4: M3a raw/grace dwell tracking (O-FIX: gated by allow_m3_dwell)
            if allow_m3_dwell:
                if in_m3a:
                    self._m3a_dwell_raw_s += dt_int
                    self._m3a_dwell_grace_s += dt_int
                    self._m3a_grace_timer_s = 0.0
                else:
                    self._m3a_dwell_raw_s = 0.0
                    grace_limit = getattr(self, "_m3a_grace_limit_s", 0.25)
                    self._m3a_grace_timer_s += dt_int
                    if self._m3a_grace_timer_s > grace_limit:
                        self._m3a_dwell_grace_s = 0.0
            else:
                self._m3a_dwell_raw_s = 0.0
                self._m3a_dwell_grace_s = 0.0
                self._m3a_grace_timer_s = 0.0

            in_m3b = (
                abs(delta_behind) <= self.th.milestones.m3b_delta_behind_tol
                and abs(delta_below) <= self.th.milestones.m3b_delta_below_tol
                and abs(delta_cross) <= self.th.milestones.m3b_delta_cross_tol
            )
            alerts.extend(
                self._update_milestone_dwell(
                    ms=self.m3b,
                    in_env=in_m3b,
                    dt_int=dt_int,
                    t_rel=t_rel,
                    dwell_req=self.th.milestones.m3b_dwell_s,
                    tag="M3B",
                    allow_dwell=allow_m3_dwell,
                )
            )
            self._update_dwell_segment(
                "m3b",
                in_m3b and allow_m3_dwell,
                t_rel,
                delta_behind,
                delta_below,
                delta_cross,
                controller_state,
            )

            # D4: M3b raw/grace dwell tracking (O-FIX: gated by allow_m3_dwell)
            if allow_m3_dwell:
                if in_m3b:
                    self._m3b_dwell_raw_s += dt_int
                    self._m3b_dwell_grace_s += dt_int
                    self._m3b_grace_timer_s = 0.0
                else:
                    # M3b exit axis logging: log which axis broke the dwell
                    if self._m3b_dwell_raw_s > 0.5:
                        _axes_failed = []
                        if abs(delta_behind) > self.th.milestones.m3b_delta_behind_tol:
                            _axes_failed.append(f"behind={delta_behind:.3f}")
                        if abs(delta_cross) > self.th.milestones.m3b_delta_cross_tol:
                            _axes_failed.append(f"cross={delta_cross:.3f}")
                        if abs(delta_below) > self.th.milestones.m3b_delta_below_tol:
                            _axes_failed.append(f"below={delta_below:.3f}")
                        alerts.append(
                            f"[M3B_EXIT] dwell={self._m3b_dwell_raw_s:.2f}s "
                            f"violated=[{', '.join(_axes_failed)}]"
                        )
                    self._m3b_dwell_raw_s = 0.0
                    self._m3b_grace_timer_s += dt_int
                    # #460 Candidate B (default-OFF => byte-identical duration-only grace).
                    if self._m3b_grace_reset(delta_behind, self._m3b_grace_timer_s):
                        self._m3b_dwell_grace_s = 0.0
            else:
                self._m3b_dwell_raw_s = 0.0
                self._m3b_dwell_grace_s = 0.0
                self._m3b_grace_timer_s = 0.0

            # Step 1a: M3a_coarse_delta — actual coarse acquisition tier (2.0/2.0/1.5, 1.0s)
            in_m3a_cd = (
                abs(delta_behind) <= self.th.milestones.m3a_coarse_delta_behind_tol
                and abs(delta_below) <= self.th.milestones.m3a_coarse_delta_below_tol
                and abs(delta_cross) <= self.th.milestones.m3a_coarse_delta_cross_tol
            )
            alerts.extend(
                self._update_milestone_dwell(
                    ms=self.m3a_coarse_delta,
                    in_env=in_m3a_cd,
                    dt_int=dt_int,
                    t_rel=t_rel,
                    dwell_req=self.th.milestones.m3a_coarse_dwell_s,
                    tag="M3A_COARSE_DELTA",
                    allow_dwell=allow_m3_dwell,
                )
            )
            self._update_dwell_segment(
                "m3a_cd",
                in_m3a_cd and allow_m3_dwell,
                t_rel,
                delta_behind,
                delta_below,
                delta_cross,
                controller_state,
            )

            # M3a_coarse_delta raw/grace dwell tracking (O-FIX: gated by allow_m3_dwell)
            if allow_m3_dwell:
                if in_m3a_cd:
                    self._m3a_cd_dwell_raw_s += dt_int
                    self._m3a_cd_dwell_grace_s += dt_int
                    self._m3a_cd_grace_timer_s = 0.0
                else:
                    self._m3a_cd_dwell_raw_s = 0.0
                    grace_limit = getattr(self, "_m3a_grace_limit_s", 0.25)
                    self._m3a_cd_grace_timer_s += dt_int
                    if self._m3a_cd_grace_timer_s > grace_limit:
                        self._m3a_cd_dwell_grace_s = 0.0
            else:
                self._m3a_cd_dwell_raw_s = 0.0
                self._m3a_cd_dwell_grace_s = 0.0
                self._m3a_cd_grace_timer_s = 0.0

        return alerts

    def _m3b_grace_reset(self, delta_behind, grace_timer_s):
        """#460 Candidate B (R7.1 metric-side): decide whether to reset the M3b GRACE dwell
        after an out-of-corridor excursion. OFF (default) => duration-only: reset iff the
        excursion has lasted longer than _m3b_grace_limit_s (byte-identical to the frozen
        baseline). ON => bounded-excursion: reset iff the excursion exceeds D seconds OR the
        |behind| magnitude exceeds M (a brief-but-large departure is NOT tolerated). Pure;
        no mutation, no control effect (observer/metric only)."""
        if getattr(self, "_m3b_excursion_tol", False):
            return (grace_timer_s > getattr(self, "_m3b_excursion_d_s", 0.5)
                    or abs(delta_behind) > getattr(self, "_m3b_excursion_mag_m", 1.0))
        return grace_timer_s > getattr(self, "_m3b_grace_limit_s", 0.25)

    def _update_dwell_segment(
        self,
        prefix: str,
        effective_in: bool,
        t_rel: float,
        delta_behind: float,
        delta_below: float,
        delta_cross: float,
        controller_state: str,
    ) -> None:
        """Track dwell segments for proof instrumentation (O-FIX).

        Args:
            prefix: "m3a", "m3b", or "m3a_cd"
            effective_in: True when in_env AND allow_dwell (strict gate)
            t_rel: current relative time
            delta_behind/below/cross: current deltas
            controller_state: current state name
        """
        seg_start = getattr(self, f"_{prefix}_seg_start")
        segments = getattr(self, f"_{prefix}_segments")

        if effective_in:
            if seg_start is None:
                # Start new segment
                setattr(self, f"_{prefix}_seg_start", t_rel)
                setattr(self, f"_{prefix}_seg_state", controller_state)
                setattr(self, f"_{prefix}_seg_max_behind", abs(delta_behind))
                setattr(self, f"_{prefix}_seg_max_below", abs(delta_below))
                setattr(self, f"_{prefix}_seg_max_cross", abs(delta_cross))
            else:
                # Update running peaks
                cur_b = getattr(self, f"_{prefix}_seg_max_behind")
                cur_z = getattr(self, f"_{prefix}_seg_max_below")
                cur_c = getattr(self, f"_{prefix}_seg_max_cross")
                setattr(self, f"_{prefix}_seg_max_behind", max(cur_b, abs(delta_behind)))
                setattr(self, f"_{prefix}_seg_max_below", max(cur_z, abs(delta_below)))
                setattr(self, f"_{prefix}_seg_max_cross", max(cur_c, abs(delta_cross)))
        else:
            if seg_start is not None:
                # Finalize segment
                segments.append(
                    DwellSegment(
                        t_start=seg_start,
                        t_end=t_rel,
                        duration_s=t_rel - seg_start,
                        max_abs_delta_behind=getattr(self, f"_{prefix}_seg_max_behind"),
                        max_abs_delta_below=getattr(self, f"_{prefix}_seg_max_below"),
                        max_abs_delta_cross=getattr(self, f"_{prefix}_seg_max_cross"),
                        state_at_entry=getattr(self, f"_{prefix}_seg_state"),
                    )
                )
                setattr(self, f"_{prefix}_seg_start", None)
                setattr(self, f"_{prefix}_seg_state", None)
                setattr(self, f"_{prefix}_seg_max_behind", 0.0)
                setattr(self, f"_{prefix}_seg_max_below", 0.0)
                setattr(self, f"_{prefix}_seg_max_cross", 0.0)

    @staticmethod
    def _update_milestone_dwell(
        *,
        ms: MilestoneState,
        in_env: bool,
        dt_int: float,
        t_rel: float,
        dwell_req: float,
        tag: str,
        allow_dwell: bool = True,
    ) -> List[str]:
        """Update milestone dwell tracking.

        O-FIX: Added allow_dwell parameter.
        - Entry detection (first_entry_s, entries count): uses in_env only (broad gate)
        - Dwell accumulation, best_dwell_s, achievement: uses in_env AND allow_dwell
        - When allow_dwell is False and in_env is True: reset _current_dwell_s (break streak)
        """
        alerts: List[str] = []

        if in_env:
            # Entry detection uses broad gate (in_env only)
            if ms.first_entry_s is None:
                ms.first_entry_s = t_rel
                ms.entries += 1
                alerts.append(f"[MILESTONE_{tag}] ENTER t={t_rel:.3f}s")

            if allow_dwell:
                # Dwell accumulation only in contact states
                ms._current_dwell_s += dt_int
                if ms._current_dwell_s > ms.best_dwell_s:
                    ms.best_dwell_s = ms._current_dwell_s

                if (not ms.achieved) and ms._current_dwell_s >= dwell_req:
                    ms.achieved = True
                    ms.first_dwell_complete_s = t_rel
                    alerts.append(
                        f"[MILESTONE_{tag}] ACHIEVED t={t_rel:.3f}s dwell={ms._current_dwell_s:.3f}s"
                    )
            else:
                # In envelope but not in contact state — break dwell streak
                ms._current_dwell_s = 0.0
        else:
            ms._current_dwell_s = 0.0

        return alerts

    # -------------------------
    # Internal: stability
    # -------------------------

    def _update_stability(
        self,
        *,
        sim_time_s: float,
        controller_state: str,
        behind_tip_m: float,
        cross_tip_m: float,
        below_tip_m: float,
    ) -> None:
        st = self.th.stability
        if controller_state in set(st.precontact_states):
            eB = behind_tip_m - st.precontact_target_behind_m
            eC = cross_tip_m - st.precontact_target_cross_m
            eZ = below_tip_m - st.precontact_target_below_m
            self._stab_pre.add(sim_time_s, eB, eC, eZ)
        if controller_state in set(st.contact_states):
            eB = behind_tip_m - st.contact_target_behind_m
            eC = cross_tip_m - st.contact_target_cross_m
            eZ = below_tip_m - st.contact_target_below_m
            self._stab_con.add(sim_time_s, eB, eC, eZ)

    # -------------------------
    # Internal: altitude
    # -------------------------

    def _update_altitude(
        self,
        *,
        sim_time_s: float,
        dt_int: float,
        controller_state: str,
        below_tip_m: float,
        follower_alt_msl: Optional[float],
    ) -> None:
        alt = self.th.altitude
        st = self.th.stability

        # Phase detection uses same groups as stability (configurable).
        in_pre = controller_state in set(st.precontact_states)
        in_con = controller_state in set(st.contact_states)

        # RELATIVE or BOTH
        if alt.mode in ("RELATIVE", "BOTH"):
            if in_pre:
                if abs(below_tip_m - alt.precontact_below_target_m) > alt.precontact_below_tol_m:
                    self.precontact_violation_time_s += dt_int
            if in_con:
                if abs(below_tip_m - alt.contact_below_target_m) > alt.contact_below_tol_m:
                    self.contact_violation_time_s += dt_int

        # ABSOLUTE or BOTH
        if alt.mode in ("ABSOLUTE", "BOTH") and follower_alt_msl is not None:
            if in_pre and alt.precontact_alt_msl_target is not None:
                if (
                    abs(follower_alt_msl - alt.precontact_alt_msl_target)
                    > alt.precontact_alt_msl_tol
                ):
                    self.precontact_violation_time_s += dt_int
            if in_con and alt.contact_alt_msl_target is not None:
                if abs(follower_alt_msl - alt.contact_alt_msl_target) > alt.contact_alt_msl_tol:
                    self.contact_violation_time_s += dt_int

    # -------------------------
    # Internal: plateau
    # -------------------------

    def _update_plateau(
        self,
        *,
        sim_time_s: float,
        dt_int: float,
        controller_state: str,
        behind_tip_m: float,
        closure_rate_mps: Optional[float],
    ) -> None:
        """Update plateau detection accumulator.

        Plateau is detected when:
        - Controller in plateau_states AND
        - behind_tip >= behind_min_m AND
        - abs(closure_rate) <= closure_rate_abs_tol_mps
        - For continuous duration >= plateau_window_s

        Uses accumulator model: time accumulates when ALL conditions met,
        resets to 0 when any condition fails.
        """
        # Already detected - nothing more to do
        if self._plateau_detected:
            return

        # Derive closure rate if not provided
        if closure_rate_mps is None:
            if self._last_behind_tip_m is not None and dt_int > 0:
                closure_rate_mps = (self._last_behind_tip_m - behind_tip_m) / dt_int
            else:
                closure_rate_mps = 0.0
        self._last_behind_tip_m = behind_tip_m

        # Store in history for to_dict() reporting
        self._closure_rate_hist.append((sim_time_s, float(closure_rate_mps), float(behind_tip_m)))

        # Trim window for history (for reporting, not detection)
        cutoff = sim_time_s - self.th.plateau.plateau_window_s
        while self._closure_rate_hist and self._closure_rate_hist[0][0] < cutoff:
            self._closure_rate_hist.popleft()

        # Check all plateau conditions
        in_plateau_state = controller_state in set(self.th.plateau.plateau_states)
        behind_ok = behind_tip_m >= self.th.plateau.behind_min_m
        closure_ok = abs(closure_rate_mps) <= self.th.plateau.closure_rate_abs_tol_mps

        all_conditions_met = in_plateau_state and behind_ok and closure_ok

        if all_conditions_met:
            # Accumulate time
            self._plateau_accum_s += dt_int

            # Check if threshold reached
            if self._plateau_accum_s >= self.th.plateau.plateau_window_s:
                self._plateau_detected = True
                self._plateau_first_detected_s = _now_rel(sim_time_s, self.t0_s)
        else:
            # Reset accumulator on any condition break
            self._plateau_accum_s = 0.0

    def _is_plateau(self) -> bool:
        """Return True if plateau has been detected.

        Plateau detection is now based on latched accumulator flag,
        NOT gated by M2 deadline.
        """
        return self._plateau_detected

    # -------------------------
    # Internal: outcome mapping
    # -------------------------

    def _map_hard_fail_to_outcome(self) -> OutcomeClass:
        # Map by latest hard-fail event.
        hard_events = [e for e in self.safety_events if e.severity == "HARD_FAIL"]
        if not hard_events:
            return OutcomeClass.FAIL
        code = hard_events[-1].code
        if code == SafetyCode.S1_OVERTAKE:
            return OutcomeClass.OVERSHOOT
        if code == SafetyCode.S2_COLLISION:
            return OutcomeClass.COLLISION
        # Everything else is generic FAIL for now.
        return OutcomeClass.FAIL


# -------------------------
# Convenience: default thresholds JSON template
# -------------------------


def write_default_thresholds_json(path: str) -> None:
    """Write a JSON template users can edit for threshold overrides."""
    th = MetricsThresholds()
    data = {
        "milestones": asdict(th.milestones),
        "safety": {**asdict(th.safety), "hard_fail_states": list(th.safety.hard_fail_states)},
        "stability": {
            **asdict(th.stability),
            "precontact_states": list(th.stability.precontact_states),
            "contact_states": list(th.stability.contact_states),
        },
        "altitude": asdict(th.altitude),
        "plateau": {**asdict(th.plateau), "plateau_states": list(th.plateau.plateau_states)},
        "dt": asdict(th.dt),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
