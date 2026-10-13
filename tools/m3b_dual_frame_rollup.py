#!/usr/bin/env python3
"""Dual-frame M3b rollup — emits side-by-side perceived + truth M3b summary.

Authored 2026-05-08 to standardize M3b reporting across both:
  - Controller-perceived (FormationSuccessMetrics, RelativePoseFusion view)
  - Boom-tip-frame raw-truth (env_method_c, SIM_STATE.lat_int + dynamic boom)

Background: through 2026-05-04 the controller-perceived tracker was the only
M3b metric in standard reports. Path F+ (2026-05-03) and Probe B+A (2026-05-04)
decomposed a +10.83 m bias (RelativePoseFusion +9.22 m, ArduPilot EKF +1.61 m)
between perceived and boom-tip-frame raw truth. This tool surfaces that
disparity in every Method C run as a categorical verdict the gate checker
and N-run anchors can grep on.

Inputs (all optional; tool degrades gracefully):
  --run-dir <path> Auto-resolve perceived/truth/paired-tick from one dir
  --controller-summary <p> Override: explicit run_summary.json (perceived)
  --env-truth <p> Override: explicit run_summary_env_truth.json
  --paired-tick <p> Override: explicit paired_tick_log_env*.jsonl
  --output <path> Required. Where to write m3b_dual_frame_summary.json

Verdict categories:
  DUAL_PASS Both frames cleared M3b STRICT (dwell ≥ 3.0 s)
  PERCEIVED_PASS_TRUTH_FAIL Controller declared M3b but truth was outside envelope
  TRUTH_PASS_PERCEIVED_FAIL Truth-frame achieved M3b but controller missed it
  DUAL_FAIL Neither frame achieved M3b
  PERCEIVED_ONLY Truth data unavailable — perceived value reported,
                             verdict deferred (legacy/historic runs)
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = "2.0"

# Reconcile M3b thresholds to the single source (formation_success_metrics.
# MilestoneThresholds), not local literals. Fall back to the equal literals only
# if tests/ is not importable (tool degrades gracefully per its docstring).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (_PROJECT_ROOT / "tests", _PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from formation_success_metrics import MilestoneThresholds  # type: ignore[import-not-found]
    _MT = MilestoneThresholds()
    M3B_DWELL_REQ_S = float(_MT.m3b_dwell_s)
    M3B_BEHIND_TOL_M = float(_MT.m3b_delta_behind_tol)
    M3B_BELOW_TOL_M = float(_MT.m3b_delta_below_tol)
    M3B_CROSS_TOL_M = float(_MT.m3b_delta_cross_tol)
    _THRESH_SOURCE = "formation_success_metrics.MilestoneThresholds"
except Exception:  # pragma: no cover - import-fallback path
    M3B_DWELL_REQ_S = 3.0
    M3B_BEHIND_TOL_M = 0.7
    M3B_BELOW_TOL_M = 0.7
    M3B_CROSS_TOL_M = 0.5
    _THRESH_SOURCE = "literal_fallback"

# Epoch-coherence gate: |median(csv_t) - median(obs_t)| above this => the two
# sim_time_s clocks are incoherent and the join cannot be trusted.
EPOCH_SKEW_MAX_S = 0.1

# Guard / DOCKING_CREEP columns surfaced from speed_arbitration.csv (boolean 0/1
# flags + string-label phase/binding columns).
_GUARD_BOOL_COLS = (
    "brake_clamp", "climb_clamp", "safety_clamp", "coupled_below_active",
    "final_approach_active", "near_precontact_active", "safety_emergency_active",
    "no_pass_violation", "vert_floor_soft_active", "vert_floor_hard_active",
    "in_precontact_corridor", "speed_override_active",
)
_GUARD_LABEL_COLS = ("state", "dc_phase", "binding_constraint")
_DOCKING_CREEP_STATE = "DOCKING_CREEP"


def _truthy(token: Optional[str]) -> bool:
    return token in ("1", "True", "true")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _resolve_paths_from_run_dir(
    run_dir: Path,
) -> Tuple[Optional[Path], Optional[Path], Optional[Path], Optional[Path]]:
    """Locate the four input artifacts inside a single run directory.

    Returns (controller_summary, env_truth, paired_tick, perceived_csv).
    perceived_csv is `speed_arbitration.csv` if present (used for the
    standalone-observer CSV-fallback path).
    """
    controller = run_dir / "run_summary.json"
    env_truth = run_dir / "run_summary_env_truth.json"
    perceived_csv = run_dir / "speed_arbitration.csv"
    paired = None
    # paired_tick_log_env<N>.jsonl — pick the first match (deterministic via sort)
    matches = sorted(run_dir.glob("paired_tick_log_env*.jsonl"))
    if matches:
        paired = matches[0]
    return (
        controller if controller.exists() else None,
        env_truth if env_truth.exists() else None,
        paired,
        perceived_csv if perceived_csv.exists() else None,
    )


# ---------------------------------------------------------------------------
# Perceived-side extraction (controller's run_summary.json)
# ---------------------------------------------------------------------------

def _extract_perceived(controller_summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Pull M3b STRICT block from FormationSuccessMetrics output."""
    out: Dict[str, Any] = {
        "available": False,
        "source": "tests/formation_success_metrics.py:MilestoneState (m3b_tight_contact)",
    }
    if controller_summary is None:
        return out
    sm = controller_summary.get("success_metrics") or {}
    ms = sm.get("milestones") or {}
    m3b = ms.get("m3b_tight_contact") or {}
    if not m3b:
        return out
    out.update({
        "available": True,
        "m3b_achieved": bool(m3b.get("achieved", False)),
        "m3b_entries": int(m3b.get("entries", 0) or 0),
        "m3b_best_dwell_s": float(m3b.get("best_dwell_s", 0.0) or 0.0),
        "m3b_first_entry_s": m3b.get("first_entry_s"),
        "m3b_first_dwell_complete_s": m3b.get("first_dwell_complete_s"),
        "outcome": sm.get("outcome"),
    })
    return out


# ---------------------------------------------------------------------------
# Truth-side extraction (env's run_summary_env_truth.json)
# ---------------------------------------------------------------------------

def _extract_truth(env_truth: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Pull M3b state from env_method_c's truth-side summary."""
    out: Dict[str, Any] = {
        "available": False,
        "source": "src/drl/env_method_c.py:_emit_truth_summary",
    }
    if env_truth is None:
        return out
    out.update({
        "available": True,
        "m3b_entries": int(env_truth.get("m3b_entries", 0) or 0),
        "m3b_best_dwell_s": float(env_truth.get("m3b_best_dwell_s", 0.0) or 0.0),
        "m3b_continuous_dwell_s_at_end": env_truth.get("m3b_continuous_dwell_s_at_end"),
        "in_m3b_at_end": env_truth.get("in_m3b_at_end"),
        "min_range_m": env_truth.get("min_range_m"),
        "min_behind_abs_m": env_truth.get("min_behind_abs_m"),
        "min_cross_abs_m": env_truth.get("min_cross_abs_m"),
        "min_below_abs_m": env_truth.get("min_below_abs_m"),
        "frame": env_truth.get("frame", "boom_tip_delta"),
        # Full truth ladder + SIM_STATE(108) health (observer schema >= 2.0).
        "milestones_truth": env_truth.get("milestones_truth"),
        "sim_state_fallback_ticks": env_truth.get("sim_state_fallback_ticks"),
    })
    return out


# Map truth-ladder milestone name -> perceived success_metrics.json milestone key.
# truth m3a_coarse uses the m3a_coarse_delta tols (2.0/1.5/2.0) => maps to the
# perceived m3a_coarse_delta_contact; truth m3a_fine -> m3a_fine_contact.
_LADDER_PERCEIVED_KEY = {
    "m1": "m1_staging",
    "m2": "m2_precontact",
    "m3a_coarse": "m3a_coarse_delta_contact",
    "m3a_fine": "m3a_fine_contact",
    "m3b": "m3b_tight_contact",
}


def _extract_perceived_ladder(
    controller_summary: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Pull each perceived milestone's {achieved,best_dwell_s,entries} from
    success_metrics.json. Missing keys -> milestone marked unavailable."""
    out: Dict[str, Dict[str, Any]] = {}
    ms: Dict[str, Any] = {}
    if controller_summary is not None:
        sm = controller_summary.get("success_metrics") or {}
        ms = sm.get("milestones") or {}
    for name, key in _LADDER_PERCEIVED_KEY.items():
        m = ms.get(key) or {}
        if m:
            out[name] = {
                "available": True,
                "achieved": bool(m.get("achieved", False)),
                "best_dwell_s": float(m.get("best_dwell_s", 0.0) or 0.0),
                "entries": int(m.get("entries", 0) or 0),
            }
        else:
            out[name] = {"available": False}
    return out


def _build_milestone_ladder(
    perceived_ladder: Dict[str, Dict[str, Any]],
    truth_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Per-milestone dual (perceived vs truth) achieved/dwell block."""
    truth_ms = (truth_summary or {}).get("milestones_truth") or {}
    ladder: Dict[str, Any] = {}
    for name in _LADDER_PERCEIVED_KEY:
        p = perceived_ladder.get(name, {"available": False})
        t = truth_ms.get(name) if isinstance(truth_ms, dict) else None
        t_avail = isinstance(t, dict)
        p_ach = bool(p.get("achieved")) if p.get("available") else None
        t_ach = bool(t.get("achieved")) if t_avail else None
        if p_ach is None or t_ach is None:
            verdict = "INCOMPLETE"
        elif p_ach and t_ach:
            verdict = "DUAL_PASS"
        elif p_ach and not t_ach:
            verdict = "PERCEIVED_PASS_TRUTH_FAIL"
        elif t_ach and not p_ach:
            verdict = "TRUTH_PASS_PERCEIVED_FAIL"
        else:
            verdict = "DUAL_FAIL"
        ladder[name] = {
            "perceived": p,
            "truth": (t if t_avail else {"available": False}),
            "verdict": verdict,
        }
    return ladder


# ---------------------------------------------------------------------------
# Disparity stats (paired_tick_log JSONL)
# ---------------------------------------------------------------------------

def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def _load_perceived_csv(
    csv_path: Path,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """Load (sim_time_s, {behind,below,cross}) from controller's
    speed_arbitration.csv. Returns parallel lists sorted by sim_time_s for
    binary-search lookup.

    Frame note: cols behind_m/below_m/cross_m (3-5) are the FRESHEST boom-tip
    perceived geometry — the same frame as the observer's raw_truth_* (the
    delta_*_m cols 86-88 are these values lagged one 50 ms tick). All three axes
    use cols 3-5 so per-axis disparity is frame-consistent.
    """
    times: List[float] = []
    axes: Dict[str, List[float]] = {"behind": [], "below": [], "cross": []}
    if not csv_path.exists():
        return times, axes
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get("sim_time_s")
                b = row.get("behind_m")
                lo = row.get("below_m")
                cr = row.get("cross_m")
                if not t or not b or not lo or not cr:
                    continue
                try:
                    t_f = float(t)
                    vals = (float(b), float(lo), float(cr))
                except (ValueError, TypeError):
                    continue
                times.append(t_f)
                axes["behind"].append(vals[0])
                axes["below"].append(vals[1])
                axes["cross"].append(vals[2])
    except OSError:
        return [], {"behind": [], "below": [], "cross": []}
    # CSV may not be strictly sorted under harness rerun edge cases; sort.
    if times:
        order = sorted(range(len(times)), key=lambda i: times[i])
        times = [times[i] for i in order]
        for k in axes:
            axes[k] = [axes[k][i] for i in order]
    return times, axes


def _nearest_index(times: List[float], sim_time_s: float) -> Tuple[Optional[int], Optional[float]]:
    """Return (index_of_nearest_sample, abs_dt) or (None, None) if times empty."""
    if not times:
        return None, None
    idx = bisect.bisect_left(times, sim_time_s)
    best_i: Optional[int] = None
    best_dt: Optional[float] = None
    for i in (idx, idx - 1):
        if 0 <= i < len(times):
            dt = abs(times[i] - sim_time_s)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_i = i
    return best_i, best_dt


def _csv_lookup_perceived(
    sim_time_s: Optional[float],
    times: List[float],
    axes: Dict[str, List[float]],
    tol_s: float,
) -> Tuple[Optional[Dict[str, float]], Optional[float], Optional[float]]:
    """Nearest-neighbor lookup of the perceived (behind,below,cross) triple by
    sim_time_s. Returns (values_dict_or_None, best_dt, matched_csv_t). values_dict
    is None when sim_time_s is missing, the CSV is empty, or the nearest sample is
    beyond tol_s; best_dt (raw nearest distance) and matched_csv_t are still
    returned so the caller can (a) distinguish "out of CSV range" from "in range
    but mis-joined" and (b) compute the signed join residual for epoch coherence.
    """
    if sim_time_s is None or not times:
        return None, None, None
    i, best_dt = _nearest_index(times, sim_time_s)
    if i is None or best_dt is None:
        return None, best_dt, None
    matched_t = times[i]
    if best_dt > tol_s:
        return None, best_dt, matched_t
    return {k: axes[k][i] for k in ("behind", "below", "cross")}, best_dt, matched_t


def _median(values: List[float]) -> Optional[float]:
    return _percentile(values, 50.0)


def _scan_paired_tick(
    path: Path, perceived_m3b_strict: bool,
    perceived_csv_path: Optional[Path] = None,
    join_tol_s: float = 0.05,
) -> Dict[str, Any]:
    """Walk paired_tick JSONL once, computing PER-AXIS disparity stats.

    `signed_d_<axis> = raw_truth_<axis> - perceived_<axis>` in the boom-tip
    frame (per Path F+ 2026-05-03 RAW-CTRL convention). Positive behind means
    raw truth is further behind the boom tip than the controller perceives —
    the controller is "ahead" of where it really is.

    Perceived source: JSONL `env_fused_<axis>_m` if present (DRL env_method_c
    case), else CSV-fallback from speed_arbitration.csv cols behind_m/below_m/
    cross_m by sim_time_s join (v98 standalone observer case).

    Coherence gates (R3/R4): emits `epoch_skew_s` = |median(csv_t)-median(obs_t)|
    and `join_violations` = # of in-CSV-range truth ticks whose nearest CSV
    sample is beyond join_tol_s (the join-tolerance gate). Both are checked by
    the caller; > thresholds => INCONCLUSIVE.
    """
    axis_names = ("behind", "below", "cross")
    out: Dict[str, Any] = {
        "available": False,
        "n_paired_ticks": 0,
        "perceived_source": None,
        "n_csv_joins": 0,
        "join_tolerance_s": join_tol_s,
        "epoch_skew_s": None,
        "join_violations": 0,
        "join_violation_samples": [],
        "raw_truth_at_perceived_m3b_strict_max_behind_m": None,
        "raw_truth_at_perceived_m3b_strict_n_ticks": 0,
    }
    for ax in axis_names:
        out[f"median_signed_d_{ax}_m"] = None
        out[f"p95_signed_d_{ax}_m"] = None
        out[f"max_abs_signed_d_{ax}_m"] = None
    if not path.exists():
        return out
    csv_times: List[float] = []
    csv_axes: Dict[str, List[float]] = {"behind": [], "below": [], "cross": []}
    if perceived_csv_path is not None:
        csv_times, csv_axes = _load_perceived_csv(perceived_csv_path)
    csv_lo = csv_times[0] if csv_times else None
    csv_hi = csv_times[-1] if csv_times else None

    deltas: Dict[str, List[float]] = {ax: [] for ax in axis_names}
    raw_at_perceived: List[float] = []
    join_residuals: List[float] = []   # obs_t - matched_csv_t over joined ticks
    n_ticks = 0
    n_in_csv_range = 0
    n_csv_joins = 0
    n_jsonl_joins = 0
    join_violations = 0
    violation_samples: List[float] = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_ticks += 1
                t = rec.get("sim_time_s")
                in_range = (
                    t is not None and csv_lo is not None
                    and csv_lo <= float(t) <= csv_hi
                )
                if in_range:
                    n_in_csv_range += 1
                raw = {ax: rec.get(f"raw_truth_{ax}_m") for ax in axis_names}
                # Perceived: prefer JSONL env_fused_*; else CSV-fallback triple.
                perc: Optional[Dict[str, float]] = None
                jsonl_perc = {ax: rec.get(f"env_fused_{ax}_m") for ax in axis_names}
                if all(jsonl_perc[ax] is not None for ax in axis_names):
                    perc = {ax: float(jsonl_perc[ax]) for ax in axis_names}
                    n_jsonl_joins += 1
                elif csv_times:
                    perc, best_dt, matched_t = _csv_lookup_perceived(
                        t, csv_times, csv_axes, join_tol_s
                    )
                    if perc is not None:
                        n_csv_joins += 1
                        if matched_t is not None:
                            join_residuals.append(float(t) - matched_t)
                    elif in_range and best_dt is not None and best_dt > join_tol_s:
                        # In CSV range but nearest sample beyond tol => mis-join.
                        join_violations += 1
                        if len(violation_samples) < 10:
                            violation_samples.append(float(t))
                if perc is not None:
                    for ax in axis_names:
                        if raw[ax] is not None:
                            deltas[ax].append(float(raw[ax]) - perc[ax])
                if perceived_m3b_strict and raw["behind"] is not None:
                    raw_at_perceived.append(abs(float(raw["behind"])))
    except OSError:
        return out
    if n_ticks == 0:
        return out
    out["available"] = True
    out["n_paired_ticks"] = n_ticks
    out["n_in_csv_range"] = n_in_csv_range
    out["n_csv_joins"] = n_csv_joins
    out["join_violations"] = join_violations
    out["join_violation_samples"] = violation_samples
    out["join_rate"] = (n_csv_joins / n_in_csv_range) if n_in_csv_range else None
    if n_csv_joins > 0 and n_jsonl_joins == 0:
        out["perceived_source"] = "csv_speed_arbitration"
    elif n_jsonl_joins > 0 and n_csv_joins == 0:
        out["perceived_source"] = "jsonl_env_fused"
    elif n_csv_joins > 0 or n_jsonl_joins > 0:
        out["perceived_source"] = "mixed"
    for ax in axis_names:
        if deltas[ax]:
            out[f"median_signed_d_{ax}_m"] = _median(deltas[ax])
            out[f"p95_signed_d_{ax}_m"] = _percentile([abs(d) for d in deltas[ax]], 95.0)
            out[f"max_abs_signed_d_{ax}_m"] = max(abs(d) for d in deltas[ax])
    # Epoch coherence from the matched-pair join residuals (obs_t - matched_csv_t),
    # NOT the difference of full-series medians (which conflates the observer's
    # longer pre-roll with a real clock offset). Bounded by join_tol for joined
    # ticks; the real coverage gate is join_violations.
    if join_residuals:
        out["epoch_skew_s"] = abs(_median(join_residuals))
    if raw_at_perceived:
        out["raw_truth_at_perceived_m3b_strict_max_behind_m"] = max(raw_at_perceived)
        out["raw_truth_at_perceived_m3b_strict_n_ticks"] = len(raw_at_perceived)
    return out


# ---------------------------------------------------------------------------
# Guard / DOCKING_CREEP correlation (speed_arbitration.csv) cross-tabbed truth
# ---------------------------------------------------------------------------

def _load_truth_timeline(
    paired_path: Optional[Path],
) -> Tuple[List[float], List[bool]]:
    """Load (sorted sim_time_s, in_m3b_truth) from paired_tick JSONL for the
    guard cross-tab. Returns empty lists when paired-tick is absent (cross-tab
    fields then degrade to None)."""
    times: List[float] = []
    in_m3b: List[bool] = []
    if paired_path is None or not paired_path.exists():
        return times, in_m3b
    try:
        with open(paired_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = rec.get("sim_time_s")
                if t is None:
                    continue
                times.append(float(t))
                in_m3b.append(bool(rec.get("in_m3b_truth", False)))
    except OSError:
        return [], []
    if times:
        order = sorted(range(len(times)), key=lambda i: times[i])
        times = [times[i] for i in order]
        in_m3b = [in_m3b[i] for i in order]
    return times, in_m3b


def _scan_guards(
    csv_path: Optional[Path],
    truth_times: Optional[List[float]] = None,
    truth_in_m3b: Optional[List[bool]] = None,
    join_tol_s: float = 0.05,
) -> Dict[str, Any]:
    """Surface safety-guard fire-counts + time-active, DOCKING_CREEP residency,
    dc_phase residency, and binding-constraint distribution from
    speed_arbitration.csv, cross-tabbed against truth-frame M3b membership when a
    truth timeline is supplied.

    Guard/DC stats need only the CSV (work for the recorded run with no
    observer). The `*_while_truth_m3b` cross-tab fields are populated only when a
    paired-tick truth timeline is present; otherwise None.
    """
    out: Dict[str, Any] = {
        "available": False,
        "truth_crosstab_available": False,
        "n_csv_rows": 0,
        "total_time_s": 0.0,
        "guards": {},
        "docking_creep": {"ticks": 0, "time_s": 0.0, "ticks_while_truth_m3b": None},
        "dc_phase_residency": {},
        "binding_constraint_dist": {},
    }
    if csv_path is None or not csv_path.exists():
        return out
    have_truth = bool(truth_times) and bool(truth_in_m3b) and len(truth_times) == len(truth_in_m3b)

    def truth_m3b_at(t: float) -> Optional[bool]:
        if not have_truth:
            return None
        i, dt = _nearest_index(truth_times, t)
        if i is None or dt is None or dt > join_tol_s:
            return None
        return truth_in_m3b[i]

    guard_ticks = {g: 0 for g in _GUARD_BOOL_COLS}
    guard_time = {g: 0.0 for g in _GUARD_BOOL_COLS}
    guard_m3b = {g: 0 for g in _GUARD_BOOL_COLS}
    dc_ticks = 0
    dc_time = 0.0
    dc_m3b = 0
    dc_phase_ticks: Dict[str, int] = {}
    dc_phase_time: Dict[str, float] = {}
    binding_dist: Dict[str, int] = {}
    n_rows = 0
    total_time = 0.0

    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader]
    except OSError:
        return out
    # Per-row dt from consecutive sim_time_s (clamped to ignore gaps/regressions).
    times: List[Optional[float]] = []
    for r in rows:
        try:
            times.append(float(r.get("sim_time_s")))
        except (TypeError, ValueError):
            times.append(None)
    for k, r in enumerate(rows):
        t = times[k]
        dt = 0.0
        if t is not None and k + 1 < len(times) and times[k + 1] is not None:
            d = times[k + 1] - t
            if 0.0 < d <= 0.5:
                dt = d
        n_rows += 1
        total_time += dt
        tm = truth_m3b_at(t) if t is not None else None
        for g in _GUARD_BOOL_COLS:
            if _truthy(r.get(g)):
                guard_ticks[g] += 1
                guard_time[g] += dt
                if tm:
                    guard_m3b[g] += 1
        state = r.get("state") or ""
        if state == _DOCKING_CREEP_STATE:
            dc_ticks += 1
            dc_time += dt
            if tm:
                dc_m3b += 1
        ph = r.get("dc_phase") or ""
        if ph:
            dc_phase_ticks[ph] = dc_phase_ticks.get(ph, 0) + 1
            dc_phase_time[ph] = dc_phase_time.get(ph, 0.0) + dt
        bc = r.get("binding_constraint") or ""
        if bc:
            binding_dist[bc] = binding_dist.get(bc, 0) + 1

    out["available"] = True
    out["truth_crosstab_available"] = have_truth
    out["n_csv_rows"] = n_rows
    out["total_time_s"] = total_time
    out["guards"] = {
        g: {
            "ticks_active": guard_ticks[g],
            "time_active_s": guard_time[g],
            "ticks_active_while_truth_m3b": (guard_m3b[g] if have_truth else None),
        }
        for g in _GUARD_BOOL_COLS
    }
    out["docking_creep"] = {
        "ticks": dc_ticks,
        "time_s": dc_time,
        "ticks_while_truth_m3b": (dc_m3b if have_truth else None),
    }
    out["dc_phase_residency"] = {
        ph: {"ticks": dc_phase_ticks[ph], "time_s": dc_phase_time[ph]}
        for ph in sorted(dc_phase_ticks)
    }
    out["binding_constraint_dist"] = dict(sorted(binding_dist.items()))
    return out


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _compute_verdict(
    perceived: Dict[str, Any], truth: Dict[str, Any]
) -> str:
    perceived_pass = bool(
        perceived.get("available")
        and perceived.get("m3b_achieved")
        and float(perceived.get("m3b_best_dwell_s") or 0.0) >= M3B_DWELL_REQ_S
    )
    if not truth.get("available"):
        return "PERCEIVED_ONLY"
    truth_pass = float(truth.get("m3b_best_dwell_s") or 0.0) >= M3B_DWELL_REQ_S
    if perceived_pass and truth_pass:
        return "DUAL_PASS"
    if perceived_pass and not truth_pass:
        return "PERCEIVED_PASS_TRUTH_FAIL"
    if truth_pass and not perceived_pass:
        return "TRUTH_PASS_PERCEIVED_FAIL"
    return "DUAL_FAIL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_summary(
    controller_summary_path: Optional[Path],
    env_truth_path: Optional[Path],
    paired_tick_path: Optional[Path],
    perceived_csv_path: Optional[Path] = None,
    tick_hz: float = 10.0,
) -> Dict[str, Any]:
    controller = _load_json(controller_summary_path) if controller_summary_path else None
    env_truth = _load_json(env_truth_path) if env_truth_path else None
    perceived = _extract_perceived(controller)
    truth = _extract_truth(env_truth)
    perceived_ladder = _extract_perceived_ladder(controller)
    milestone_ladder = _build_milestone_ladder(perceived_ladder, truth)

    # Join tolerance derived from the observer tick rate (R5): 0.5/tick_hz =
    # half a truth-tick width (default 10 Hz -> 0.05 s).
    join_tol = 0.5 / float(tick_hz) if tick_hz > 0 else 0.05

    perceived_pass = bool(
        perceived.get("available")
        and perceived.get("m3b_achieved")
        and float(perceived.get("m3b_best_dwell_s") or 0.0) >= M3B_DWELL_REQ_S
    )
    effective_paired = paired_tick_path if (paired_tick_path and paired_tick_path.exists()) else None
    if effective_paired is None and env_truth is not None and env_truth.get("paired_tick_log_path"):
        ept = Path(env_truth["paired_tick_log_path"])
        if ept.exists():
            effective_paired = ept
    if effective_paired is not None:
        disparity = _scan_paired_tick(
            effective_paired, perceived_pass,
            perceived_csv_path=perceived_csv_path, join_tol_s=join_tol,
        )
    else:
        disparity = {"available": False, "n_paired_ticks": 0}

    # Guard / DOCKING_CREEP scan (CSV-only; cross-tab needs the truth timeline).
    truth_times, truth_in_m3b = _load_truth_timeline(effective_paired)
    guards = _scan_guards(perceived_csv_path, truth_times, truth_in_m3b, join_tol)

    verdict = _compute_verdict(perceived, truth)

    # Coherence / health gates (R3/R4/C2). Any populated reason => INCONCLUSIVE.
    inconclusive: List[str] = []
    epoch_skew = disparity.get("epoch_skew_s")
    if epoch_skew is not None and epoch_skew > EPOCH_SKEW_MAX_S:
        inconclusive.append("INCONCLUSIVE_EPOCH_SKEW")
    if int(disparity.get("join_violations") or 0) > 0:
        inconclusive.append("INCONCLUSIVE_JOIN_TOLERANCE_EXCEEDED")
    fb = truth.get("sim_state_fallback_ticks")
    if fb is not None and int(fb) > 0:
        inconclusive.append("INCONCLUSIVE_SIM_STATE_FALLBACK")

    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/m3b_dual_frame_rollup.py",
        "threshold_source": _THRESH_SOURCE,
        "thresholds": {
            "frame": "boom_tip_delta",
            "behind_max_m": M3B_BEHIND_TOL_M,
            "below_max_m": M3B_BELOW_TOL_M,
            "cross_max_m": M3B_CROSS_TOL_M,
            "dwell_req_s": M3B_DWELL_REQ_S,
        },
        "coherence": {
            "tick_hz": float(tick_hz),
            "join_tolerance_s": join_tol,
            "epoch_skew_s": epoch_skew,
            "epoch_skew_max_s": EPOCH_SKEW_MAX_S,
            "join_violations": int(disparity.get("join_violations") or 0),
            "sim_state_fallback_ticks": fb,
        },
        "inputs": {
            "controller_summary": str(controller_summary_path) if controller_summary_path else None,
            "env_truth": str(env_truth_path) if env_truth_path else None,
            "paired_tick": str(effective_paired) if effective_paired else None,
            "perceived_csv": str(perceived_csv_path) if perceived_csv_path else None,
        },
        "perceived": perceived,
        "truth": truth,
        "milestone_ladder": milestone_ladder,
        "disparity": disparity,
        "guards": guards,
        "verdict": verdict,
        "inconclusive": inconclusive,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Emit dual-frame (perceived + truth) M3b summary."
    )
    ap.add_argument("--run-dir", type=Path, default=None,
                    help="Directory containing run_summary.json + "
                         "run_summary_env_truth.json + paired_tick_log_env*.jsonl + "
                         "speed_arbitration.csv (CSV-fallback for standalone runs)")
    ap.add_argument("--controller-summary", type=Path, default=None)
    ap.add_argument("--env-truth", type=Path, default=None)
    ap.add_argument("--paired-tick", type=Path, default=None)
    ap.add_argument("--perceived-csv", type=Path, default=None,
                    help="speed_arbitration.csv (CSV-fallback when JSONL "
                         "lacks env_fused_*; standalone observer case)")
    ap.add_argument("--output", type=Path, required=True,
                    help="Where to write m3b_dual_frame_summary.json")
    ap.add_argument("--tick-hz", type=float, default=None,
                    help="Observer tick rate Hz; join_tolerance_s = 0.5/tick_hz "
                         "(default OBSERVER_TICK_HZ env or 10.0)")
    args = ap.parse_args(argv)

    controller_path = args.controller_summary
    env_truth_path = args.env_truth
    paired_path = args.paired_tick
    perceived_csv_path = args.perceived_csv
    if args.run_dir is not None:
        c, t, p, pcsv = _resolve_paths_from_run_dir(args.run_dir)
        if controller_path is None:
            controller_path = c
        if env_truth_path is None:
            env_truth_path = t
        if paired_path is None:
            paired_path = p
        if perceived_csv_path is None:
            perceived_csv_path = pcsv

    tick_hz = args.tick_hz
    if tick_hz is None:
        try:
            tick_hz = float(os.environ.get("OBSERVER_TICK_HZ", "10.0"))
        except (TypeError, ValueError):
            tick_hz = 10.0

    summary = build_summary(
        controller_path, env_truth_path, paired_path,
        perceived_csv_path=perceived_csv_path, tick_hz=tick_hz,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    inconclusive = summary.get("inconclusive") or []
    # One-line stdout for harness/grep convenience.
    print(
        f"[m3b_dual_frame] verdict={summary['verdict']} "
        f"perceived_dwell={summary['perceived'].get('m3b_best_dwell_s')} "
        f"truth_dwell={summary['truth'].get('m3b_best_dwell_s')} "
        f"inconclusive={','.join(inconclusive) if inconclusive else 'none'} "
        f"out={args.output}"
    )
    # Non-zero exit when a coherence/health gate trips (R3/R4/C2).
    return 3 if inconclusive else 0


if __name__ == "__main__":
    sys.exit(main())
