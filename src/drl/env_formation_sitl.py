"""Gymnasium environment wrapping a single ArduPilot SITL instance.

Each ``FormationSITLEnv`` manages:
  - An ArduPilot SITL process (via ``start_formation_sitl_fixed.sh --direct``)
  - A ``leader_broadcast_fixed.py`` process
  - Warmup via ``WarmupController`` (baseline subclass) to reach M2 precontact
  - 4Hz DRL decisions (20Hz sub-ticks × ACTION_REPEAT=5) with residual action

Port isolation is achieved via ``HARNESS_PORT_OFFSET = env_id * PORT_STRIDE``.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import os
import random
import signal
import subprocess
import statistics as st
from collections import deque
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Phase E Batch 3a: MRRL raw component list for per-episode telemetry
# ---------------------------------------------------------------------------
# Components whose max / min / mean are logged per-episode for pathology
# diagnostics. Set is the RAW (pre-weight) component set actually accumulated
# in self._reward_accumulators at reset time. Telemetry is passive - no
# reward path reads from these min/max fields.
#
# Note: the operator M6 summary of this scope (see PHASE_G_IMPLEMENTATION_
# REPORT.md) cited 15 named components including r_terminal and r_passthrough.
# The actual env-side raw-reward accumulator tracks 16 different components:
# r_terminal / r_passthrough are accumulated only in their _weighted forms,
# while r_range_penalty / r_exit / r_overspeed (env-side weighted-only
# components) DO appear as raw tracked values. Telemetry mirrors the actual
# accumulator contents to stay data-consistent; the discrepancy with M6's
# verbal list is documented in the implementation report.
_MRRL_RAW_COMPONENTS = [
    'r_ref', 'r_prox', 'r_dwell', 'r_contact_dwell',
    'r_safe', 'r_smooth', 'r_wake',
    'r_m3a_coarse', 'r_m3a_fine',
    'r_range_penalty', 'r_approach', 'r_action_rate',
    'r_exit', 'r_overspeed', 'r_barrier', 'r_hold_fwd',
    'r_precontact',   # R1ELECT-703 (AMENDMENT R1)
]

import logging

from . import config
from . import flight_state_recorder
from . import gps_ladder
from ._process_utils import pdeathsig_preexec
from .observation_builder import (
    EnvelopeFlags,
    FollowerState,
    LeaderState,
    ObservationBuilder,
    SensorInfo,
)
from .residual_action import BaselineCommand, ResidualActionProcessor
from .reward_mrrl import (MRRLRewardComputer, ceiling_shaping_reward,
                          nopass_shaping_reward)

# Baseline imports (from tests/ via __init__.py path setup)
from geometry import GeometryCalculator, RelativeGeometry
from formation_control_laws import FormationControlLaws
from formation_envelope import EnvelopeChecker
import formation_config as fc
from sim_time_authority import SimTimeAuthority
from state_machine import ControllerState

try:
    from wake_turbulence_model import WakeTurbulenceZone
except ImportError:
    WakeTurbulenceZone = None  # type: ignore[misc,assignment]


# Path to the SITL launcher and leader broadcaster
_TESTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "tests",
)
_SITL_SCRIPT = os.path.join(_TESTS_DIR, "start_formation_sitl_fixed.sh")
_LEADER_SCRIPT = os.path.join(_TESTS_DIR, "leader_broadcast_fixed.py")
# JSON-lockstep external FDM (#309 P1; launched ONLY when FD_TRAIN_BACKEND=json)
_FDM_PY = os.path.join(
    os.path.dirname(_TESTS_DIR), "tools", "json_fdm", "plane6dof_fdm.py",
)

logger = logging.getLogger(__name__)

# FD-ENV-MAVLINK-DIAG: env-side MAVLink relay diagnostic gate.
# When set truthy, _send_commands emits WARNING-level diagnostic logs
# around all command emission branches (sends, exceptions, bounded
# summaries). Diagnostic-only — no behavior change. Used for Arm 2
# wrapper-vs-controller causality adjudication after Arm 1 manual
# harness reached Verdict E (mechanical-null via SimTimeAuthority Q-gate).
_FD_ENV_MAVLINK_DIAG = os.environ.get(
    "FD_ENV_MAVLINK_DIAG", "0"
).strip().lower() in ("1", "true", "yes", "on")

# FIX-α (#328): forced repositions re-arm on the interval limit — 22.6 Hz forcing
# realizes ~0 climb, standalone's ~7 Hz climbs+docks (#327 DREAD_REPORT).
# Default-ON since #329; explicit "0" = legacy.
_FD_ENV_REPOSITION_FORCE_REARM = os.environ.get(
    "FD_ENV_REPOSITION_FORCE_REARM", "1").strip() == "1"
from .reposition_cadence import force_send_allowed  # noqa: E402

# #329: env speed-only while C++ engaged (see drl_controller_ref parity block).
_FD_DRLREF_POSTTICK_PARITY = os.environ.get(
    "FD_DRLREF_POSTTICK_PARITY", "0").strip() == "1"

# #331 DC L1-nav fix: in DC precision states, withhold the H5
# GUIDED_CHANGE_HEADING re-arm so target_heading_type stays NONE (DO_REPOSITION
# already cleared it, GCS_MAVLink_Plane.cpp:629) and ArduPlane uses calc_nav_roll
# L1 cross-track nav (mode_guided.cpp:74-75) — matching the frozen baseline, which
# sends DO_REPOSITION only. RCA #331: the yaw-hold branch (mode_guided.cpp:48-70)
# bypasses L1, stranding cross-nulling on the weak -0.175*cross bias -> slow
# lateral runaway (measured 0.8->9->-112 m; harness holds cross ~0 with L1).
# Default-ON since #441 (operator adoption 2026-07-15 on the B1 matched-seed A/B,
# 4e06ccd4: lever CAUSAL 6/6 — ON parks |cross| <= 0.052 m vs OFF +2.9..+3.1 m
# bilateral cycle); explicit "0" = legacy yaw-hold re-arm in DC states.
_FD_ENV_DC_L1_NAV = os.environ.get("FD_ENV_DC_L1_NAV", "1").strip() == "1"
# DC precision states where L1 must own lateral (state_machine.ControllerState).
_DC_L1_STATES = frozenset({"ALT_CAP_HOLD_RANGE", "DOCKING_CREEP", "HOLD_CONTACT"})
# #337 Contract A (default-ON since #339, adopted on the #337+#338 N=8 flight
# evidence; kill-switch FD_ENV_RAW_TRUTH_ALIGN2=0 yields the ORIGINAL
# receive-side pairing instrument — the superseded #334 v1 receive-side align
# path and its env flag were DELETED in #342, Option E): align the leader to
# the follower SIM_STATE's SENDER-SIDE sample time (= the co-emitted ATTITUDE
# tbms, probe-measured adjacency p95 0.08 ms) instead of any receive-side
# clock. The follower truth position is NEVER modified (epoch = the follower
# sample itself). Fixes the #336 sample-interleave (leader pinned per 50 ms
# quantum while SIM_STATE advances mid-quantum => +/-1.24 m sawtooth,
# age-decorrelated). Gate INDEPENDENT of LEADER_EXTRAP_ENABLE by design.
_FD_ENV_RAW_TRUTH_ALIGN2 = os.environ.get("FD_ENV_RAW_TRUTH_ALIGN2", "1").strip() == "1"
# T-391 warmup acceptance gate.
# The frozen controller declares M2/HOLD_PRECONTACT on FSM state ENTRY, which
# can happen while the follower is still converging: measured across 33
# D1/D1a legs the scored window opened +2.97 m median AFT of the 10 m station
# with 26/33 outside the +/-1.5 m outer tolerance (START_OFFSET_CENSUS.json).
# M2's own criterion lives in R7.1 and is NOT touched; instead the env
# verifies the controller's OWN perception geometry at warmup handoff and
# fails closed into the EXISTING warmup-retry loop when it is out of station.
# PERCEPTION-SIDE ONLY: reads ctrl.get_geometry(), never _raw_truth_geom/rm_*
# -- G4 Tier-0 holds.
#
# SHAPE = WAIT-AND-VERIFY. The 2026-08-10 smoke
# measured the FIRST live handoff at behind_err +2.914 m -- essentially the
# +2.97 m historical median -- so the offset is SYSTEMATIC, not a boot
# lottery. A refuse-on-sight gate would therefore reject ~100 % of legs and
# exhaust all 5 warmup retries (FM7, pre-declared 20 % escalation threshold:
# fired at 1/1); the leg is not broken, it is merely EARLY. So the env now
# flies the frozen baseline's own commands for a bounded settle window and
# opens the scored window once the geometry is actually on station.
#
# RETRACTED (2026-08-10, source audit): the prior claim that ticking here is
# unsafe because tick_once() -> _update_success_metrics() mutates
# episode-scoped state. The success-metrics clock is latched ONCE at
# _reset_sim_timers_after_pass (baseline_formation_controller.py:3089), which
# runs at the timebase PASS -- i.e. it is ALREADY running throughout warmup on
# every leg. Ticking a few more seconds continues an in-progress observer, it
# does not start one. The real hazard is the opposite one, documented at
# _boot_screen_classify: tick_once() in capture mode does NOT transmit, so a
# ticked window would fly UNCOMMANDED. The settle loop therefore transmits the
# captured baseline command verbatim (zero-residual passthrough) and runs at
# the END of _run_warmup, after self._mav is bound and target-pinned.
#
# Enforcement default OFF until the 6-leg validation smoke grades it; the
# smoke arms it with FD_WARMUP_ACCEPT_GATE=1. Disarmed => no ticking, no
# transmitting, single snapshot only == byte-identical to the pre-T-391 path.
_FD_WARMUP_ACCEPT_GATE = os.environ.get(
    "FD_WARMUP_ACCEPT_GATE", "0").strip() == "1"
# Settle budget in SIM seconds and the consecutive in-station sample count that
# opens the window. 60 s is the contract budget; 3 samples (~0.15 s at 20 Hz)
# rejects a single-sample glance through the box without waiting out a
# meaningful part of the leg.
_WARMUP_SETTLE_BUDGET_S = float(
    os.environ.get("FD_WARMUP_SETTLE_BUDGET_S", "60").strip())
_WARMUP_SETTLE_DWELL_N = int(
    os.environ.get("FD_WARMUP_SETTLE_DWELL_N", "3").strip())


def _parse_gate_below_tol(raw):
    """GATETOL-746: GATE-LOCAL below tolerance.

    ★ WHY THIS IS NOT A formation_config EDIT. The obvious implementation --
    lower PRECONTACT_BELOW_OUTER_TOL_M from 4.0 to 2.0 -- was REFUSED: that
    constant is consumed by the FROZEN baseline FSM through
    EnvelopeChecker.is_in_precontact_outer_envelope
    (tests/baseline_formation_controller.py:6938), so lowering it globally
    would change R7.1 behaviour through the back door and silently re-tier the
    HOLD/REJOIN logic. The election was about the GATE's station box, so the
    override lives on the gate and nowhere else.

    Unset => None => the formation_config value, byte-identical to the
    pre-election gate. A MALFORMED value RAISES rather than falling back: a
    gate flown at 4.0 while the launch record says 2.0 is exactly the silent
    corruption this project keeps paying for.
    """
    if not raw:
        return None
    v = float(raw)                      # ValueError propagates deliberately
    if not 0.0 < v <= 50.0:
        raise ValueError(
            f"FD_WARMUP_GATE_BELOW_TOL_M out of range: {v!r}")
    return v


_FD_WARMUP_GATE_BELOW_TOL_M = _parse_gate_below_tol(
    os.environ.get("FD_WARMUP_GATE_BELOW_TOL_M", "").strip())


def _gate_below_tol_m(fc):
    """The below tolerance the GATE must use (override, else the fc tier)."""
    if _FD_WARMUP_GATE_BELOW_TOL_M is not None:
        return _FD_WARMUP_GATE_BELOW_TOL_M
    return float(fc.PRECONTACT_BELOW_OUTER_TOL_M)
# Which FSM states mean "keep waiting" vs "stop waiting". MEASURED 2026-08-10
# on smoke leg 90011: the FSM demoted HOLD_PRECONTACT -> REJOIN_PRECONTACT one
# second after declaring M2, precisely BECAUSE the aircraft was 3.45 m out of
# station. REJOIN_PRECONTACT sits BEFORE HOLD_PRECONTACT in the nominal
# progression (state_machine.py:23-33) -- it is the controller flying itself
# back, which is exactly the thing worth waiting for. The first draft treated
# any state change as forward progress and opened the window after 1.0 s; the
# smoke caught it. Forward states end the wait (real docking progress); the
# exception/terminal states end it WITHOUT refusing (the episode owns those).
_SETTLE_BAND_STATES = ("HOLD_PRECONTACT", "REJOIN_PRECONTACT")
_SETTLE_FORWARD_STATES = ("ALTITUDE_CAPTURE", "ALT_CAP_HOLD_RANGE",
                          "DOCKING_CREEP", "HOLD_CONTACT", "SUCCESS")
# T-71: extend ALIGN2-class
# leader alignment to the ENV channel. _compute_geometry pairs the follower
# GPS position sample (epoch = follower _last_position.time_boot_ms) with the
# leader wire sample AS RECEIVED (epoch = leader last_state.time_boot_ms) —
# an unaligned 0-50 ms pairing that paints a +/-1.2 m along-track square wave
# at broadcast cadence (24.7 m/s x 50 ms = 1.235 m; MEASURED fresh-vs-stale
# p50 gap 1.28 m, sd 1.23 == v*dt). The RAW channel has been aligned since
# #337 and ctrl since #496 B-path; env was the sole unaligned consumer — and
# it feeds obs ch0-9, the reward's in_m3b gate, AND _check_termination, so
# this flag is BEHAVIORAL (not observability-only like the raw align).
# ADOPTED default-ON 2026-07-27 ((a) on AB2_ALIGN2_VERDICT.md).
# Evidence: the mechanism fingerprint — env behind-bias organized by leader age —
# was reproduced by EVERY flag-OFF leg and absent from EVERY flag-ON leg across
# two condition classes: age-gap 1.30 -> 0.04 m at CEP 0.25 (v1) and
# 1.28 -> -0.002 m at CEP 0.05 (v2, age-flat through 150 ms). DISCLOSED
# DEVIATION: the v2 registered sd-ratio bar read 2.98x against a 3.00x
# requirement (0.7% short) — that bar was a conductor-derived engineering
# threshold whose implied noise floor (<=0.451 m) landed within 3 mm-of-sd of
# the channel's measured physical floor (0.454 m at CEP 0.05), so it could not
# discriminate at that margin; adoption rests on the co-primary mechanism bar,
# not the sd ratio. Kill-switch: FD_ENV_ALIGN2=0 restores the legacy
# receive-side pairing byte-identically.
_FD_ENV_ALIGN2 = os.environ.get("FD_ENV_ALIGN2", "1").strip() == "1"
# #475: align2 COHERENT leader
# snapshot — operand + DR velocity + stamp from ONE atomically-replaced
# LeaderState (last_state is REPLACED per update, baseline_formation_
# controller.py:1469; a single object grab is coherent under the GIL). The
# pre-fix block mixed self._leader_* (an earlier grab) with a fresh
# last_state stamp; a one-leader-period straddle mis-measures age2 by
# 50-100 ms => dead-reckoning error of k*1.236 m painted onto raw_behind
# (the T0 fail-leg fingerprint — rcadwell Phase B, d8e9533f). Raw channel
# is control-inert (see :4215) => no control-path effect. Kill-switch =0
# restores the pre-fix operand reads with identical computed values.
_FD_ENV_ALIGN2_COHERENT = os.environ.get("FD_ENV_ALIGN2_COHERENT", "1").strip() == "1"
# #475 R4 (2026-07-17): kinematic tick-snap stamp synthesis for unstamped
# SIM_STATE samples (the mode-1 hold-release painter). Independent kill-switch
# so R1/R2/R3 (ALIGN2_COHERENT) and R4 are separately revertible.
_FD_ENV_ALIGN2_STAMP_SYNTH = os.environ.get(
    "FD_ENV_ALIGN2_STAMP_SYNTH", "1").strip() == "1"
# T-228 — DEFAULT OFF.
# MEASURED: on FAULT boots SITL splits the ATTITUDE/SIM_STATE co-emission, so
# ATTITUDE arrives ~33-42 ms BEFORE its SIM_STATE (CLEAN: d_s ~ 0, same burst)
# while `time_boot_ms` keeps advancing correctly. The 5 ms wall-clock window
# then REJECTS a stamp whose VALUE is right, pushing the sample into R4
# synthesis — mode-5 share reaches 88.3% on FAULT legs. The window validates
# pairing by ARRIVAL TIMING, a proxy that breaks precisely on the boots we
# care about.
# FIX: widen the window under this flag so a correct-but-early stamp survives
# to the EXISTING R3 kinematic gate (`_simstate_pair_implausible`), which
# validates by CONTENT (|dpos - v*dt|, ~0 correct vs ~1.236 m mis-paired) and
# already runs on the result; R5 re-stamps anything R3 rejects. No new
# machinery, no new data path — the JSON lockstep clock is NOT reachable from
# this process (env only launches the FDM; it never reads that socket), which
# is why fix design v1's option A was withdrawn.
# 0.060 covers the measured 33-42 ms split with margin while staying inside
# one 50 ms tick + jitter. Correctness does NOT rest on this number: R3 is the
# arbiter, the window only decides what R3 is allowed to see.
_FD_ENV_ALIGN2_WIDE_PAIR = os.environ.get(
    "FD_ENV_ALIGN2_WIDE_PAIR", "0").strip() == "1"
_FD_ENV_ALIGN2_PAIR_WINDOW_S = 0.060 if _FD_ENV_ALIGN2_WIDE_PAIR else 0.005
# T-240 —
# DEFAULT OFF. Root cause [source-verified]: ATTITUDE is requested at 20 Hz
# (50 ms interval) but SIM_STATE(108) at 10 Hz (100 ms), so ArduPilot's GCS
# scheduler puts them in DIFFERENT interval buckets; a bucket's phase is
# frozen from millis16() at allocation (GCS_Common.cpp:1812-1815) and never
# re-syncs => per-boot-random, run-stable relative phase in [0,50) ms — the
# measured 33-42 ms ATTITUDE-early split on FAULT boots. FIX: re-request 108
# at ATTITUDE's exact 50,000 us interval on the SAME link (last write wins in
# the per-link registry) => exact-interval match joins ATTITUDE's bucket =>
# within-bucket ascending-enum order (MSG_ATTITUDE=3 < MSG_SIM_STATE=34)
# emits the pair back-to-back every pass (measured 0.19 ms median co-arrival
# at matched 20 Hz, uwb_truth_feed_delivery RCA) => the 5 ms pairing window
# always passes and R4/R5 fallbacks starve. Scoring-path only: GPI(33) and
# SIMSTATE(164) rates are NOT touched (UWB horizontal truth stays 164@10 Hz,
# baseline-identical; A1/Hunk-RR rejected for frozen-baseline comparability).
_FD_ENV_SIMSTATE_RATE_MATCH = os.environ.get(
    "FD_ENV_SIMSTATE_RATE_MATCH", "0").strip() == "1"
_RATE_MATCH_MSG_ID = 108           # SIM_STATE
_RATE_MATCH_INTERVAL_US = 50000    # == ATTITUDE 20 Hz bucket (exact match)
_RATE_MATCH_WITNESS_N = 11         # first 10 inter-arrivals => one log line
# A2-r2. The r1 arming block BLOCKED the env
# for 1.0 s at the warmup->episode handoff (`for _ in range(2): send;
# sleep(0.5)` — and it slept after the LAST send too, so half of it bought
# nothing). G-1 MEASURED the consequence on all 24 banked A/B legs: the
# capture-start sim gap is 1.048-1.053 s on 12/12 ON legs vs 0.050 s on 12/12
# OFF; the MAVLink stream flies straight through it (max gap 1.2-1.3x median),
# so the world moved ~1 s while the env neither observed nor commanded; the
# first captured tick then reads truth-side behind -10.7..-16.1 m (IN FRONT)
# against a true ~+13 m, a 24-29 m discontinuity; and the +5 s recovery state
# rank-separates the four legs that early-terminated EXACTLY (four lowest of
# twelve, p=1/495). r2 keeps every property r1's comment claims — lands after
# the controller's [5/7] 10 Hz request, before the first paired sample, with a
# ~0.5 s-spaced redundancy send against boot-time loss — and removes ONLY the
# blocking: send once immediately, then re-send from the step path once the
# deadline passes. Same total send count, same wire shape, no thread (a timer
# would race the env's own writes on the same link).
_RATE_MATCH_RESEND_DELAY_S = 0.5
# Escape hatch: restore the exact r1 blocking shape without a revert, so the
# two revisions stay A/B-able. Default OFF => r2 is what flies.
_FD_ENV_RATE_MATCH_LEGACY_SLEEP = os.environ.get(
    "FD_ENV_RATE_MATCH_LEGACY_SLEEP", "0").strip() == "1"
# #391: the emitter->fusion delivery diagnostic
# (uwb_fusion_diag.csv) is opened and closed inside the FROZEN controller's
# run(), which on THIS assembly spans WARMUP ONLY — run()'s finally closes the
# writer at the M2 handoff, before any docking state exists, and the env then
# drives the controller via tick_once() with _uwb_diag_writer=None, so every
# later row is silently skipped. Diagnosed 2026-07-12 (moderca_r2 AMENDMENT 2),
# ledgered as T-1 "#391 NEEDS GO", and it went on to cost T-240 its D5
# criterion (T-279: 25 legs header-only over 11,879 in-state ticks). The fix is
# LIFECYCLE CALLS to the controller's OWN R7.1 methods — zero controller edits,
# zero new fields, zero new reads, LOGGING-ONLY, and byte-identical when the
# flag is unset. Import-time read of the SAME env var the controller gates on
# (bfc:256), so the two cannot disagree within a process.
_FD_UWB_FUSION_DIAG = os.environ.get(
    "FD_UWB_FUSION_DIAG", "").strip() == "1"
# F-RAW: the sim-state truth-feed sender can
# latch its stamp one 50 ms lockstep tick ahead of the leader stream at boot
# (rm_foll_tbms - rm_lead_tbms == +50 ms for the whole leg, seed-correlated).
# ALIGN2 then dead-reckons the leader one tick forward => truth geometry
# displaced by v_ground*50 ms (1.24 m @ 24.7 m/s; T92_FAULT_RCA.md). This
# flag arms a LEG-LEVEL compensation: estimate the epoch offset over the
# first _REPOCH_N valid stamp pairs (median snapped to the 50 ms grid) and
# subtract it INSIDE the _age2 computation ONLY — meta/CSV stamps stay RAW
# so the offline T-93 witness still measures the wire truth. Unstable
# estimates FAIL CLOSED (rm_repoch_state=-1, uncompensated, logged). Truth
# channel is observability-only (control-inert, :4215) => no control-path
# effect either way. Default OFF: adoption requires the T92_FAULT_RCA.md
# validation battery (unit fixtures + OFF-path legacy-column equivalence +
# clean-verified re-fly of a fault-prone cohort).
_FD_RAW_EPOCH_COMP = os.environ.get("FD_RAW_EPOCH_COMP", "0").strip() == "1"
_REPOCH_N = 32           # estimation window size (valid stamp pairs)
_REPOCH_B_START = 200    # window B starts after this many valid pairs
_REPOCH_TICK_MS = 50.0   # lockstep leader broadcast quantum (ms)
# frv AMBER lesson (Entry 27): a SINGLE boot window latches on TRANSIENT
# offsets (real for ~32 pairs, gone after) and over-compensates the leg.
# v2 latches only when TWO time-separated windows (pairs 1-32 and 201-232,
# ~12 s apart) agree on the same snapped offset; disagreement or
# instability FAILS CLOSED (state -1, marked, never guessed). A post-latch
# rolling guard demotes to -1 if the raw median walks off the latched
# value (transients longer than window B).
_RAW_ALIGN_MAX_AGE_S = 0.25  # cap the leader dead-reckon advance (s)
# FIXA (2026-08-18, operator-authorized adoption battery, flag-gated default
# OFF): the OBREFUTE-692/D2ARCA-690 census showed the raw-vs-aligned age is a
# TWO-SIDED 50 ms lattice whose per-leg CENTER (0 or -50 ms, boot-phase draw)
# is stable while the per-row value jitters +/-1 tick; the `0 < age` guard
# then one-sidedly dead-reckons the positive half of symmetric quantisation
# noise (55% of D2a panel legs quarantined). FIXA subtracts a rolling-median
# center snapped to the tick grid and dead-reckons ONLY genuine staleness
# (centered age >= 1 tick). Offline replay over all 43 banked D2+D2a legs:
# 43/43 CLEAN, corrections retained on 25-40% of rows (FIXA_REPLAY.json).
# The RAW age keeps flowing to meta/CSV so the offline T-93 witness still
# measures the wire truth (the F-RAW recording principle).
_FD_RAW_ALIGN_CENTER = os.environ.get(
    "FD_RAW_ALIGN_CENTER", "0").strip() == "1"
_RAW_ALIGN_TICK_S = 0.05          # lockstep broadcast quantum (s)
_RAW_ALIGN_CENTER_WIN = 64        # rolling-median window (rows)
_RAW_ALIGN_R_EARTH_M = 6371000.0  # match GeometryCalculator.latlon_to_ned convention

# T-165 EPOCH PROBE (T-93 arc; contract T93_PROBE_CONTRACT.md, +
# election (A) 2026-07-31). Observability ONLY: settles the one open T-93
# question — WHICH side picks the wrong ATTITUDE tick (broadcaster stale-hold
# vs pairer ATT(k+1) accept). Default OFF => the added branches are a single
# boolean test; nothing is written back anywhere. ON: bounded log lines only
# (seam: first _EPOCH_PROBE_N emits/episode; pairer: per-process budget).
_FD_EPOCH_PROBE = os.environ.get("FD_EPOCH_PROBE", "0").strip() == "1"
_EPOCH_PROBE_N = 40            # age-seam lines per episode
_EPOCH_PROBE_PAIRER_BUDGET = 200   # pairer lines per process
# T-392b (aligner adoption of the epoch stamp) was REMOVED 2026-08-10 after its
# premise was REFUTED in flight: across 70,890 T-394 smoke rows the wire stamp
# equalled the broadcaster's raw latch on 70,876 (99.98%), and all 14 exceptions
# differed by 1 ms, not the 50 ms tick the hypothesis required. There was
# nothing for the substitution to correct. The T-93 one-tick boot split is
# REAL but arises elsewhere -- it appeared on 2/6 smoke legs with adoption
# armed and inert. T-392a EMISSION (below) is KEPT: it is the instrument that
# produced this refutation. See SUBSTRATE_FIXES_CONTRACTS.md amendment SF-A3.


class _EpochProbePairerBudget:
    """Mutable per-process budget for the pairer probe line (module-level so
    the module-level pairer helper can decrement it; reset only by tests)."""
    left = _EPOCH_PROBE_PAIRER_BUDGET


def _probe_pairer(d_s, att_tbms, window_s):
    """T-165: log one pairer-decision observation (accept/reject derivable
    from d_s vs [-0.001, window_s] by the offline reader). NEVER raises —
    a probe that can kill a leg is worse than no probe (E4)."""
    try:
        if _EpochProbePairerBudget.left <= 0:
            return
        _EpochProbePairerBudget.left -= 1
        logger.info("[EPOCH_PROBE] pairer d_s=%.6f att_tbms=%s window_s=%.4f",
                    float(d_s), int(att_tbms), float(window_s))
    except Exception:
        pass

# Dispatch P — T0-SL runtime taint probe (goals-v4 G4; PROBE_SPEC). Default-OFF
# => byte-identical (every probe branch is guarded by self._t0sl_probe = False).
# When FD_T0SL_PROBE=1: each step the observability-only truth geom is canary-
# poisoned and obs/reward are asserted canary-free (any truth leak into the
# policy surface becomes a large/non-finite signature). Runtime backstop to the
# static leakage_static_check (evidence, not proof, at rest).
_FD_T0SL_PROBE = os.environ.get("FD_T0SL_PROBE", "0").strip() == "1"
_T0SL_CANARY = 1.0e6  # canary magnitude injected into truth caches in probe mode

# T-25 (2026-07-22): flag-gated training boot screen. Default OFF => byte-
# identical. ON => reset() classifies the boot post-warmup with the EXACT
# #513-validated union rule and rerolls a firing boot BEFORE any transition can
# exist (the hook sits before _open_paired_csv and before the first step()), so
# a fired boot can never reach the replay buffer by construction.
# The three decision constants are PINNED to tools/b2_screen.py — the env must
# not import from tools/ at runtime, so the values are mirrored here and
# tests/unit/test_train_boot_screen.py asserts equality field-by-field; drift
# fails the suite. Estimand caveat (T25_PRECONDITION.md): the certified
# population under screening is CLEAN-BOOT CONDITIONAL — firing predicts LOW
# dwell (median 0.25 s vs 15.15 s, r=-0.524), so screened rates are biased UP
# vs all-attempts rates; the disclosure below states this mechanically.
_BOOT_SCREEN_AGE2_THRESH_MS = 25.0   # == b2_screen.AGE2_THRESH_MS
_BOOT_SCREEN_BIAS_THRESH_M = 1.0     # == b2_screen.BIAS_THRESH_M (signed; T-32 owns fix-or-retire)
_BOOT_SCREEN_MIN_MODE0_FRAC = 0.20   # == b2_screen.MIN_MODE0_FRACTION
_BOOT_SCREEN_TICKS = 60              # ~3 s passive pump-only sampling window
_BOOT_SCREEN_MAX_REROLLS = 3         # per external reset; cap-exhausted boots are KEPT + flagged
_BOOT_SCREEN_HALT_MIN_ATTEMPTS = 10  # HALT-fraction denominators below this never HALT

# T-29 O1 (2026-07-22): the orphan-ArduPlane pause/resume used
# `pkill -f serial0=tcp:<port>`, one subprocess per pause AND per resume.
# MEASURED on this box: 15.9 ms median / 33.6 ms max per call (the /proc scan
# alone — pattern matching nothing), i.e. ~32 ms of blocking latency added to
# EVERY training step. Because sim tracks wall 1:1, the plant legitimately
# flies further before the env's next read, inflating sub-tick dt (F-2 bar B6,
# see O1_CAUSAL_RESULT.md). Resolving the PID ONCE with a pure-/proc scan and
# then using os.kill (~0.5 us) removes that cost. `pkill` remains the fallback
# whenever the PID cannot be resolved, so behaviour is unchanged in the worst
# case — only the delivery path is faster.


def _cmdline_of(pid: int) -> Optional[str]:
    """Space-joined /proc cmdline for PID, or None if unreadable."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
    except (OSError, IOError):
        return None


def _exe_of(pid: int) -> Optional[str]:
    """Resolved executable path for PID (/proc/<pid>/exe), or None."""
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except (OSError, IOError):
        return None


def _find_arduplane_pid(cmd_needle: str, exe_name: str = "arduplane") -> Optional[int]:
    """PID of the ONE live ArduPlane whose cmdline contains CMD_NEEDLE.

    Identity is established by **/proc/<pid>/exe** — the resolved binary — NOT
    by cmdline text. This is not a stylistic preference:

      MEASURED 2026-07-22. A cmdline-substring scan is NOT identifying. Even
      requiring BOTH "arduplane" and "serial0=tcp:<port>" still matched a
      `/bin/bash -c ...` process, because the shell's own script text mentioned
      both strings — the scan selected that shell and a real SIGSTOP was
      delivered to it during unit testing. Any process can put arbitrary text
      in its argv; only the exe link says what a process actually IS.

    Returns None on no match, AMBIGUOUS match (>1), or error; callers MUST
    treat None as "fall back to pkill" (legacy behaviour, unchanged).
    """
    try:
        me = os.getpid()
        hits = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == me:
                continue
            exe = _exe_of(pid)
            if not exe or os.path.basename(exe) != exe_name:
                continue
            cmd = _cmdline_of(pid)
            if cmd and cmd_needle in cmd:
                hits.append(pid)
                if len(hits) > 1:
                    return None  # ambiguous -> fail closed
        return hits[0] if hits else None
    except Exception:
        return None


def _t0sl_leak(obs, reward, canary=_T0SL_CANARY):
    """True iff obs/reward carry a canary-magnitude or non-finite signature —
    the runtime leak detector (PROBE_SPEC Probe 1). Pure; unit-tested."""
    thresh = canary / 2.0
    mo = float(np.max(np.abs(obs))) if getattr(obs, "size", 0) else 0.0
    return (not bool(np.all(np.isfinite(obs)))) or mo >= thresh or abs(float(reward)) >= thresh


def _center_age(age_s, window):
    """FIXA: append the RAW age to the rolling window, subtract the
    grid-snapped rolling median (the leg's lattice center), and return an
    EFFECTIVE age that dead-reckons only genuine staleness: centered age in
    [1 tick, cap] passes through, anything else (sub-tick jitter, negative,
    over-cap) returns 0.0 so `_align_leader_sample`'s `0 < age` guard
    no-ops. Mirrors the FIXA_REPLAY rule exactly. Pure apart from the
    window append; never mutates its input value."""
    window.append(age_s)
    center = _RAW_ALIGN_TICK_S * round(
        st.median(window) / _RAW_ALIGN_TICK_S)
    eff = age_s - center
    if _RAW_ALIGN_TICK_S <= eff <= _RAW_ALIGN_MAX_AGE_S:
        return eff
    return 0.0


def _align_leader_sample(lat, lon, vn, ve, age_s):
    """#334: dead-reckon a stale leader lat/lon forward by velocity*age (raw-truth
    time-alignment). Returns inputs UNCHANGED when age is non-positive (negative =
    clock skew) or exceeds the cap (pathological staleness). Pure; no truth field."""
    if not (0.0 < age_s <= _RAW_ALIGN_MAX_AGE_S):
        return lat, lon
    lat_a = lat + math.degrees(vn * age_s / _RAW_ALIGN_R_EARTH_M)
    lon_a = lon + math.degrees(
        ve * age_s / (_RAW_ALIGN_R_EARTH_M * math.cos(math.radians(lat))))
    return lat_a, lon_a


def _raw_step_implausible(prev_behind, prev_t, new_behind, now_t,
                          max_step_m=0.8, max_dt_s=0.06):
    """#475 R2: TRUE relative along-track motion cannot exceed ~16 m/s (a
    >0.8 m step within one <=0.06 s tick; worst measured breakaways <=~8 m/s
    => <=0.4 m/tick). Such a step marks the truth row PAIRING-INCONSISTENT
    (residual epoch stutters surviving the R1 single-grab fix — post-R1 jump
    census 45/58/239 vs the <10 bar). Pure predicate; observability-side only."""
    if prev_behind is None or prev_t is None or now_t is None:
        return False
    dt = now_t - prev_t
    if not (0.0 < dt <= max_dt_s):
        return False
    return abs(new_behind - prev_behind) > max_step_m


def _coherent_leader_operands(lrs, fb_lat, fb_lon, fb_vn, fb_ve):
    """#475: single-grab leader operands for the align2 raw-truth pairing.
    Returns (lat, lon, vn, ve) from ONE LeaderState object so position, DR
    velocity, and the caller's stamp read cannot straddle a feed update
    (LeaderState is replaced per update => object field reads are coherent).
    Falls back to the caller's cached fields when lrs is None. Pure; no
    truth field; observability-side only."""
    if lrs is None:
        return fb_lat, fb_lon, fb_vn, fb_ve
    return (getattr(lrs, "lat", fb_lat), getattr(lrs, "lon", fb_lon),
            getattr(lrs, "vn_ms", fb_vn), getattr(lrs, "ve_ms", fb_ve))


def _leader_probe_env(base_env, log_dir):
    """BOOTPROBE-ON: the T-186 boot-draw
    probe defaults ON at the spawn seam. Every D2a leg could only INFER a
    correct leader spawn because no leg had the probe armed; this injects
    FD_LEADER_BOOT_PROBE=1 into the CHILD environment only, so the
    broadcaster stays byte-identical and an explicit =0 (or any non-empty
    value) still wins. FD_LEADER_PROBE_DIR is set explicitly when unset
    because BootProbeSink falls back to cwd (".") -- the $HOME-landing
    trap. Pure; returns a new dict; never mutates base_env."""
    lenv = dict(base_env)
    if lenv.get("FD_LEADER_BOOT_PROBE", "").strip() == "":
        lenv["FD_LEADER_BOOT_PROBE"] = "1"
    if lenv.get("FD_LEADER_PROBE_DIR", "").strip() == "":
        lenv["FD_LEADER_PROBE_DIR"] = log_dir
    return lenv


def _align_age_from_tbms(follower_tbms, leader_tbms):
    """#336: raw-truth pairing age from SIM-TIME boot stamps (the true leader
    staleness the env already measures as ``leader_age_ms`` = follower_tbms -
    leader_msg_tbms). Returns ``(follower - leader)/1000`` seconds when BOTH
    stamps are valid (>0), else 0.0 — the sentinel makes ``_align_leader_sample``
    no-op via its ``0 < age`` guard. Fixes #335 SG-1: the #334 wall-receive-time
    delta read ~0 (leader + follower pumped in the same cycle), so the align was
    inert live. Pure; observability-only; consumes no truth field."""
    if follower_tbms is None or leader_tbms is None:
        return 0.0
    if follower_tbms <= 0 or leader_tbms <= 0:
        return 0.0
    return (follower_tbms - leader_tbms) / 1000.0


def _repoch_estimate(deltas):
    """F-RAW (T-92): boot-window epoch estimate. ``deltas`` = the first
    _REPOCH_N valid (att_tbms - lead_tbms) values in ms. Returns
    ``(median, mad, snapped, action)``; ``snapped`` = median snapped to the
    50 ms grid; ``action``: "none" (snapped 0 — the healthy-boot case,
    measured 0 ms on every clean leg), "comp" (stable non-zero offset,
    subtract it), "failclosed" (non-zero but MAD > one tick — mark the leg,
    never guess; contract T92_FAULT_RCA.md item 3). Pure; unit-tested with
    synthetic +/-1-tick fixtures."""
    s = sorted(deltas)
    n = len(s)
    med = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    dev = sorted(abs(d - med) for d in s)
    mad = dev[n // 2] if n % 2 else 0.5 * (dev[n // 2 - 1] + dev[n // 2])
    snapped = int(round(med / _REPOCH_TICK_MS) * _REPOCH_TICK_MS)
    if snapped == 0:
        return med, mad, snapped, "none"
    if mad > _REPOCH_TICK_MS:
        return med, mad, snapped, "failclosed"
    return med, mad, snapped, "comp"


def _simstate_pair_implausible(prev_lat, prev_lon, prev_tbms,
                               lat, lon, tbms, vn, ve,
                               tol_m=0.6, accel_allow=1.5, max_gap_s=2.0):
    """#475 R3: follower-side pairing-consistency acceptance (the symmetric
    twin of the R1 coherent leader grab). A SIM_STATE position mis-paired with
    an off-by-one adjacent-ATTITUDE stamp makes the align2 age wrong by one
    tick => an exact v*0.05 = 1.236 m level shift in raw_behind (SG-2 census:
    leader feed exonerated, every jump at the R2 release tick). Test: vector
    residual |dpos - v_wire*dt| vs the last ACCEPTED pair — ~0 for every
    correctly paired sample REGARDLESS of gap length, ~1.236 m for every
    mis-pair, so persistent desyncs never re-enter (an implied-speed ratio
    test re-admits them after 3 ticks; this form does not). tol grows by
    accel_allow*dt^2 so real maneuvering over longer gaps never trips it
    (3 m/s^2 class vs a 1.236 m quantum). Pure; returns (reject, resid_m).
    Liveness: bootstrap/no-stamp/NaN-velocity/long-gap all ACCEPT — only a
    positive, testable inconsistency rejects (Contract-A)."""
    if prev_lat is None or prev_lon is None:
        return False, 0.0          # bootstrap: first sample always accepted
    if prev_tbms is None or prev_tbms <= 0 or tbms is None or tbms <= 0:
        return False, 0.0          # unstamped: existing mode-1 hold covers it
    dt = (tbms - prev_tbms) / 1000.0
    if dt <= 0.0:
        return True, 0.0           # non-monotonic follower stamp: mis-pair
    if dt > max_gap_s:
        return False, 0.0          # starvation escape: re-lock on long gaps
    if vn != vn or ve != ve:       # NaN wire velocities: cannot test
        return False, 0.0
    m_per_deg = 111320.0
    dn = (lat - prev_lat) * m_per_deg
    de = (lon - prev_lon) * m_per_deg * math.cos(math.radians(lat))
    resid = math.hypot(dn - vn * dt, de - ve * dt)
    return resid > (tol_m + accel_allow * dt * dt), resid


def _simstate_snap_stamp(prev_lat, prev_lon, prev_tbms, lat, lon, vn, ve,
                         k_max=40, tol_m=0.6, accel_allow=1.5, tol_cap=1.1):
    """#475 R4: kinematic tick-snap stamp synthesis for UNSTAMPED SIM_STATE.

    The mode-1 hold path (att_tbms<=0 => reuse previous aligned geometry) was
    measured as the last #475 painter: heavy-mode boots draw 18.9% unstamped
    samples in 3-7-tick runs, and every hold RELEASE lands the accumulated
    leader catch-up as a k x 1.236 m level shift (SG-2 run-2 census: 261/261
    jumps at provenance-hold boundaries). Wall clocks are unusable here
    (wire = sender EMIT clock, 77.2% exact-tick; row counting 87.2%) — but
    the position payload itself is tick-faithful: at cruise one 50 ms tick
    == 1.236 m >> noise, so Delta-pos vs the last ACCEPTED anchor snaps to
    the true tick count (0 ambiguous across 959 measured runs; counterfactual
    reconstruction kills 261/3/0 -> 0/0/0 jumps, sg2_run2_census_r4.py).

    Returns the synthesized att_tbms (anchor + 50*k_star ms) or None to
    DECLINE (caller falls through to the legacy mode-1 hold): no anchor, NaN
    velocity, or best residual above tolerance. tol grows accel_allow*dt^2
    but is CAPPED below one quantum (tol_cap < 1.236) so the argmin stays
    unique at every k; longer/odder gaps decline (no worse than today).
    Never called for stamped samples (caller gates on att_tbms<=0); a
    synthesized candidate still passes the R3 acceptance gate downstream.
    """
    if (prev_lat is None or prev_lon is None
            or prev_tbms is None or prev_tbms <= 0):
        return None                    # bootstrap: no anchor yet
    if vn != vn or ve != ve:
        return None                    # NaN wire velocities: cannot snap
    m_per_deg = 111320.0
    dn = (lat - prev_lat) * m_per_deg
    de = (lon - prev_lon) * m_per_deg * math.cos(math.radians(lat))
    best_k, best_r = 0, float("inf")
    for k in range(1, k_max + 1):
        dt = 0.05 * k
        r = math.hypot(dn - vn * dt, de - ve * dt)
        if r < best_r:
            best_k, best_r = k, r
    dt = 0.05 * best_k
    if best_r > min(tol_m + accel_allow * dt * dt, tol_cap):
        return None                    # off-grid position: decline to hold
    return prev_tbms + 50 * best_k


def _reinit_uwb_diag_after_warmup(ctrl):
    """#391: re-open the dock-state UWB delivery diag the controller's run()
    closed at the warmup handoff. No-op unless FD_UWB_FUSION_DIAG=1 AND the
    writer is actually closed (so a live handle is never truncated). Calls the
    controller's own R7.1 method — a CALL, not an edit; LOGGING-ONLY. Never
    raises into warmup: a diagnostic that can abort a leg is worse than a
    missing diagnostic. Returns True iff the writer is open afterwards."""
    if not _FD_UWB_FUSION_DIAG or ctrl is None:
        return False
    if getattr(ctrl, "_uwb_diag_writer", None) is not None:
        return True                       # already open — do not re-open ('w')
    try:
        ctrl._init_uwb_fusion_diag()
    except Exception:                                         # noqa: BLE001
        logger.warning("[UWB_DIAG] post-warmup re-init FAILED — diag stays "
                       "empty for this leg (leg itself unaffected)",
                       exc_info=True)
        return False
    return getattr(ctrl, "_uwb_diag_writer", None) is not None


def _close_uwb_diag_pre_teardown(ctrl):
    """#391 mirror of run()'s finally: flush + close the re-opened diag while
    the controller is still alive. Never raises into teardown (a raise here
    must not skip the SIGTERM/SIGKILL loop, nor mislabel the leg)."""
    if not _FD_UWB_FUSION_DIAG or ctrl is None:
        return False
    try:
        ctrl._close_uwb_fusion_diag()
    except Exception:                                         # noqa: BLE001
        logger.warning("[UWB_DIAG] pre-teardown close FAILED — rows may be "
                       "unflushed", exc_info=True)
        return False
    return True


def _send_sim_state_interval(mav,
                             msg_id=_RATE_MATCH_MSG_ID,
                             interval_us=_RATE_MATCH_INTERVAL_US):
    """T-240: request SIM_STATE at ATTITUDE's exact interval (rate-match fix).

    Mirrors baseline_formation_controller.request_message_rate's wire shape
    (MAV_CMD_SET_MESSAGE_INTERVAL=511 via command_long_send) but targets
    sys=1/comp=1 explicitly (the ENV_TARGET_PIN invariant) and takes the
    interval directly in microseconds so the bucket match is EXACT — the
    scheduler joins buckets only on exact interval equality; any mismatch
    allocates a separate bucket with a frozen random phase (the defect).
    Returns True when the send call succeeded (an ACK witness is collected
    passively by the caller); False on any exception (send is best-effort,
    the D1 pcap/rate witnesses adjudicate delivery).
    """
    try:
        mav.mav.command_long_send(
            1, 1,                    # follower autopilot (ENV_TARGET_PIN)
            511,                     # MAV_CMD_SET_MESSAGE_INTERVAL
            0,                       # confirmation
            float(msg_id),           # param1: message ID
            float(interval_us),      # param2: interval (us)
            0.0, 0.0, 0.0, 0.0, 0.0  # unused params
        )
        return True
    except Exception:
        return False


def _latest_new_sim_state(mav, last_stamp):
    """Passive SIM_STATE read from pymavlink's last-message cache.

    FIX-1 2026-07-05 (#320 RCA): returns (lat, lon, alt_m, wire_stamp,
    att_tbms) from ``mav.messages['SIM_STATE']`` WITHOUT consuming wire bytes
    (the F1-alt-1 pattern, cf. the GPI passive read in _pump_mavlink). Returns
    None when no SIM_STATE is cached, the cache is unreadable, or the cached
    message's parse timestamp equals ``last_stamp`` (already observed — no
    re-count). Parsing preserved from the legacy drain: lat_int/lon_int
    int32/1e7 PRIMARY (float lat/lon fields KNOWN BROKEN under SITL,
    project_option_c_phase36a); float fallback last resort.

    #337 Contract A: ``att_tbms`` = the SENDER-SIDE sim sample time of this
    SIM_STATE, captured from the co-emitted ATTITUDE in the same TCP burst
    (probe-measured adjacency p95 0.08 ms at matched 20 Hz; window 5 ms).
    Sentinel -1 when no adjacent ATTITUDE (counted by the caller).
    """
    try:
        msg = mav.messages.get('SIM_STATE', None)
    except Exception:
        return None
    if msg is None:
        return None
    stamp = getattr(msg, '_timestamp', None)
    if stamp is None or stamp == last_stamp:
        return None
    lat_int = getattr(msg, 'lat_int', None)
    lon_int = getattr(msg, 'lon_int', None)
    if lat_int is not None and lon_int is not None:
        lat, lon = lat_int / 1e7, lon_int / 1e7
    else:
        # Float fallback (known unreliable; last resort).
        lat = float(getattr(msg, 'lat', 0.0))
        lon = float(getattr(msg, 'lon', 0.0))
    att_tbms = _adjacent_attitude_tbms(mav, stamp)
    # #442: also surface the TRUE SIM_STATE wire velocities (vn/ve/vd) — the
    # ACTUAL truth velocities, NOT the GPI/EKF estimate. Parsed-and-discarded
    # until now; used only for the paired-CSV rm_* operand-provenance columns
    # (offline de-ZOH instrument input, #443/#444). Additive: never touches the
    # geometry math or any training surface.
    return (lat, lon, float(getattr(msg, 'alt', 0.0)), stamp, att_tbms,
            float(getattr(msg, 'vn', float('nan'))),
            float(getattr(msg, 've', float('nan'))),
            float(getattr(msg, 'vd', float('nan'))))


def _adjacent_attitude_tbms(mav, sim_state_stamp,
                            window_s=_FD_ENV_ALIGN2_PAIR_WINDOW_S):
    """#337 Contract A: sender-side sim sample time for a SIM_STATE via its
    co-emitted ATTITUDE (same GCS scheduler pass => adjacent in the TCP burst;
    probe-measured p95 0.08 ms, and ATTITUDE precedes SIM_STATE in 1199/1199
    probe batches). Accept ONLY an ATTITUDE parsed BEFORE (or within 1 ms of)
    the SIM_STATE and within ``window_s`` — a burst that drains into the NEXT
    batch leaves ATTITUDE(k+1) in the cache with a LATER parse stamp than
    SIM_STATE(k); accepting it mis-stamps by +50 ms (the B1v2 first-cut
    regression). Sentinel -1 otherwise. Pure read; only a timestamp."""
    try:
        att = mav.messages.get('ATTITUDE', None)
        if att is None:
            return -1
        a_stamp = getattr(att, '_timestamp', None)
        if a_stamp is None:
            return -1
        _d = float(sim_state_stamp) - float(a_stamp)  # >0: ATT before SIM
        if _FD_EPOCH_PROBE:   # T-165 observability; helper never raises
            _probe_pairer(_d, getattr(att, 'time_boot_ms', -1) or -1, window_s)
        if _d < -0.001 or _d > window_s:
            return -1
        tbms = int(getattr(att, 'time_boot_ms', -1) or -1)
        return tbms if tbms > 0 else -1
    except Exception:
        return -1


def _prep_feed_sim_clock(sta, msg) -> bool:
    """FIX-2 2026-07-05 (#319 RCA): feed a message's follower ``time_boot_ms``
    into ``sta`` (a local SimTimeAuthority) so preposition holds can be gated
    on sim time. Returns True iff a valid time_boot_ms was fed. Any message
    carrying time_boot_ms (ATTITUDE / SYSTEM_TIME / GLOBAL_POSITION_INT stream
    by default on SERIAL1 at ~1 Hz each) is accepted — no stream-rate change is
    requested. Pure clock feed: no metric is read back here."""
    tbms = getattr(msg, 'time_boot_ms', None)
    if tbms is None:
        return False
    try:
        return bool(sta.update_from_mavlink(int(tbms)))
    except Exception:
        return False


def check_m3b(behind_err: float, cross_err: float, below_err: float) -> bool:
    """Stateless instantaneous M3b membership check (band/precontact variant).

    Returns True if the relative position errors are within the tight
    precontact envelope (±0.7 m behind, ±0.5 m cross, ±0.7 m below) used
    by DRL ``hold_m3b`` reward shaping. This is the same logic the
    :class:`FormationSITLEnv` uses internally; the env's ``_check_m3b``
    method delegates here.

    Exposed at module level (C3X Intervention 3) so external callers —
    notably the ``QValueDistributionCallback`` in :mod:`drl.callbacks` —
    can compute bucket assignments from replay-buffer transitions
    without instantiating an env. Body is byte-equivalent to the prior
    in-class implementation.
    """
    return (
        abs(behind_err) < 0.7
        and abs(cross_err) < 0.5
        and abs(below_err) < 0.7
    )


def check_contact_m3b(
    behind_m: float, cross_m: float, below_m: float,
) -> bool:
    """Stateless instantaneous M3b membership check (contact/boom-tip variant).

    Returns True if the boom-tip-relative geometry is within the True
    FlightDock M3b envelope (``±CONTACT_M3B_BEHIND_TOL`` behind,
    ``±CONTACT_M3B_CROSS_TOL`` cross, ``±CONTACT_M3B_BELOW_TOL`` below).
    Tolerances are read from :mod:`drl.config`.

    This is the same logic :meth:`FormationSITLEnv._check_contact_m3b`
    uses internally; the env method delegates here. Exposed at module
    level (C3X IC4 baseline rework) so the
    ``QValueDistributionCallback`` can compute the contact-variant
    bucket assignments from replay-buffer transitions without
    instantiating an env. Body is byte-equivalent to the prior in-class
    implementation, including the inclusive ``<=`` boundary semantics
    (note: the band-variant uses strict ``<``; this asymmetry is
    preserved verbatim).
    """
    return (
        abs(behind_m) <= config.CONTACT_M3B_BEHIND_TOL
        and abs(below_m) <= config.CONTACT_M3B_BELOW_TOL
        and abs(cross_m) <= config.CONTACT_M3B_CROSS_TOL
    )


def _repo_logs_dir() -> str:
    """The repo-root ``logs/`` the FDM streams are written to (see _launch_fdm)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "logs")


def _bank_next_life_dir(bank_dir: str) -> Optional[str]:
    """Create the next never-used ``life_NNNN`` dir under ``bank_dir``.

    Sequence is scan-max+1 and creation is EXCLUSIVE (``exist_ok=False``), so a
    bank can never overwrite a lifetime it already holds — that would put
    evidence destruction inside the evidence-preservation fix. Racing workers
    bump instead of colliding. Returns None if no free slot was claimed.
    """
    seq = 0
    if os.path.isdir(bank_dir):
        for name in os.listdir(bank_dir):
            if name.startswith("life_") and name[5:].isdigit():
                seq = max(seq, int(name[5:]) + 1)
    for bump in range(8):
        cand = os.path.join(bank_dir, "life_%04d" % (seq + bump))
        try:
            os.makedirs(cand, exist_ok=False)
            return cand
        except FileExistsError:
            continue
    return None


def _bank_fdm_wind_extract(bank_dir: str, log_dir: str, env_id: int,
                           decim: Optional[int] = None) -> Optional[str]:
    """T-286: distil the FDM stream that is about to be TRUNCATED.

    plane6dof_fdm.py opens --log-jsonl with mode "w" and the env relaunches the
    FDM from inside reset(), so every episode destroyed the realized-turbulence
    record (twice observed). This copies out only the two fields the sigma
    witness parses, every ``decim``-th record, under the CANONICAL filename
    s1417_sigma_witness_v2 already matches — so no consumer changes. 10 Hz was
    MEASURED to move mean sigma by -0.000085 on 0.278 (0.03%, ~100x inside the
    0.010 tolerance) while cutting 688 MB to ~1.3 MB. The full-rate original is
    left untouched: net disk growth is zero.

    NEVER RAISES — reset()'s warmup-retry ``except`` catches OSError, so an
    unguarded failure here would masquerade as a launch failure and burn every
    retry. Returns a witness string, or None when nothing was banked.
    """
    import json as _json  # matches the file's established local-import pattern
    try:
        if decim is None:
            decim = int(os.environ.get("FD_FDM_BANK_DECIM", "120") or "120")
        decim = max(1, int(decim))
        src = os.path.join(log_dir, "fdm_env%s.jsonl" % env_id)
        if not os.path.isfile(src) or os.path.getsize(src) <= 0:
            return None
        dst_dir = _bank_next_life_dir(bank_dir)
        if dst_dir is None:
            return None
        dst = os.path.join(dst_dir, "fdm_env%s.jsonl" % env_id)
        kept = 0
        with open(src) as fin, open(dst, "x") as fout:
            fout.write(_json.dumps({"t286_decim": decim, "src": src}) + "\n")
            for i, line in enumerate(fin):
                if i % decim:
                    continue
                try:
                    rec = _json.loads(line)
                except Exception:
                    continue
                t, w = rec.get("fdm_timestamp"), rec.get("wind_ned")
                if t is None or not w or len(w) != 3:
                    continue
                fout.write(_json.dumps({"fdm_timestamp": t,
                                        "wind_ned": w}) + "\n")
                kept += 1
        return "[T286_BANK] env%s -> %s kept=%d decim=%d" % (
            env_id, dst, kept, decim)
    except Exception as exc:      # forensics must NEVER fail a flight
        logger.warning("[T286_BANK] skipped env%s: %s", env_id, exc)
        return None


def _bank_if_enabled(env_id: int, log_dir: Optional[str] = None) -> None:
    """Bank one FDM lifetime iff FD_FDM_BANK_DIR is set; otherwise INERT.

    Unset is the default everywhere (all existing campaigns, the F.2 launcher),
    and with it unset this function touches nothing at all.
    """
    bank = os.environ.get("FD_FDM_BANK_DIR", "").strip()
    if not bank:
        return
    witness = _bank_fdm_wind_extract(
        bank, log_dir if log_dir is not None else _repo_logs_dir(), env_id)
    if witness:
        logger.info("%s", witness)


class WallClockStallDetector:
    """Detect frozen sim-time by comparing wall-clock advance to sim-time advance.

    If ``time.monotonic()`` advances by ``threshold_s`` without any change in
    ``current_sim_time``, the detector fires — indicating that SITL has
    crashed or become unresponsive.

    During SIGSTOP pauses, ``time.monotonic()`` does NOT advance in the
    stopped worker process, so the detector naturally ignores pause periods.

    Parameters
    ----------
    threshold_s: float
        Wall-clock seconds without sim-time advance before declaring stall.
    """

    def __init__(self, threshold_s: float = 30.0) -> None:
        self.threshold_s = threshold_s
        self._last_sim_time: float = 0.0
        self._last_wall_time: float = 0.0

    def reset(self) -> None:
        """Reset at episode start."""
        self._last_sim_time = 0.0
        self._last_wall_time = time.monotonic()

    def bump(self) -> None:
        """Reset wall-clock anchor without touching sim-time reference.

        Call after SIGCONT to prevent false stall detection caused by
        wall-clock advancing during on-policy training update (PPO) while SITL was SIGSTOPped.
        """
        self._last_wall_time = time.monotonic()

    def check(self, current_sim_time: float) -> bool:
        """Return True if sim-time appears stalled.

        Should be called once per sub-tick after ``_pump_mavlink()``.
        """
        now = time.monotonic()
        if current_sim_time != self._last_sim_time:
            # Sim-time advanced — reset wall-clock anchor
            self._last_sim_time = current_sim_time
            self._last_wall_time = now
            return False
        # Sim-time unchanged — check wall-clock elapsed
        wall_elapsed = now - self._last_wall_time
        return wall_elapsed >= self.threshold_s


class FormationSITLEnv(gym.Env):
    """Gymnasium environment for formation docking via SITL.

    Parameters
    ----------
    env_id: int
        Instance ID (0, 1, 2, ...). Determines port offsets.
    speedup: float
        SITL speedup factor (1.0 or 2.0).
    episode_duration_s: float
        Maximum DRL-phase sim-time per episode.
    wind_speed_ms: float
        Wind speed for leader broadcaster.
    wind_turb: float
        Wind turbulence intensity.
    curriculum_manager: optional
        Reference to CurriculumManager for domain randomisation.
    warmup_max_sim_s: float
        Maximum sim-time for warmup phase (baseline controller to M2).
    """

    metadata = {"render_modes": []}

    # T-455: class-level defaults for the per-episode telemetry accumulators.
    # step() reads these before writing them, and the established test idiom
    # builds partial-mock envs via __new__ (bypassing __init__), so without a
    # class default every such mock raises AttributeError the moment a new
    # accumulator is added. All four are immutable scalars, so there is no
    # shared-mutable-default hazard: the first assignment in __init__/reset/
    # step shadows them with an instance attribute. Declaring them here also
    # makes the contract explicit — these always exist, and they are telemetry.
    _max_above_boom_fused_m: float = 0.0
    _max_above_boom_truth_m: float = 0.0
    _min_boom_tip_range_truth_m: float = float('inf')
    _guard_fires_episode: int = 0
    _guard_fires_base: Optional[int] = None

    def __init__(
        self,
        env_id: int = 0,
        speedup: float = 2.0,
        episode_duration_s: float = config.EPISODE_DURATION_S,
        wind_speed_ms: float = 0.0,
        wind_turb: float = 0.0,
        curriculum_manager=None,
        warmup_max_sim_s: float = config.WARMUP_MAX_SIM_S,
        ui_mode: str = "none",
        wind_channels: bool = True,
        seed: Optional[int] = None,
        *,
        out_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.env_id = env_id
        self.speedup = speedup
        self.episode_duration_s = episode_duration_s
        self.wind_speed_ms = wind_speed_ms
        self.wind_turb = wind_turb
        self.curriculum_manager = curriculum_manager
        self._warmup_max_sim_s = warmup_max_sim_s
        self._ui_mode = ui_mode
        self._wind_channels = wind_channels

        # SMOKE-A1 EP_DIR contract: out_dir is the per-invocation root;
        # reset() computes per-episode subdir at <out_dir>/episode_NNN_envM/.
        self._out_dir: Optional[str] = out_dir
        self._episode_index: int = 0
        self._current_ep_dir: Optional[str] = None
        self._current_ep_idx: Optional[int] = None

        # Phase G seed chain (worker-local seeding for SubprocVecEnv forkserver
        # children). __init__ runs in the worker process, so this seeds the
        # worker-local random/numpy RNG used by curriculum.sample_domain_
        # randomization(). _first_reset_done gates super().reset(seed=) so the
        # env's np_random is initialized from the stored seed on the first
        # reset only (Gymnasium convention).
        self._initial_seed: Optional[int] = seed
        self._first_reset_done: bool = False
        # D1 (FD_DR_SEED_V2): when ON, DR uses a private per-episode Generator
        # built in reset() (no global reseed → no SB3 global-RNG contamination).
        # When OFF, the legacy worker-local global seeding runs verbatim.
        self._dr_seed_v2: bool = os.environ.get("FD_DR_SEED_V2", "0") == "1"
        self._episode_rng = None
        self._episode_weather_seed: Optional[int] = None
        # #465: per-episode FDM DrydenWind seed word (A11: independent per
        # episode AND per env). Set every reset(); consumed by _launch_fdm()
        # ONLY when the FD_FDM_WIND_* config block is present.
        self._episode_wind_seed: Optional[int] = None
        if seed is not None and not self._dr_seed_v2:
            random.seed(seed)
            np.random.seed(seed)

        # Port computation
        self._port_offset = env_id * config.PORT_STRIDE
        self._follower_port = config.PORT_BASE + self._port_offset
        self._leader_port = config.PORT_BASE_LEADER + self._port_offset
        self._timesync_port = config.PORT_BASE_TIMESYNC + self._port_offset

        # #309 P1: SITL physics backend selector. Default "native" (--model
        # quadplane) is byte-identical to pre-P1. "json" swaps to the external
        # JSON-lockstep FDM (--model JSON) + spawns plane6dof_fdm.py. Every
        # json-only code path below is gated on self._backend.
        self._backend = os.environ.get("FD_TRAIN_BACKEND", "native").strip().lower()
        self._fdm_port = config.FDM_PORT_BASE + self._port_offset

        # T-29 (2026-07-22): JSON pair-freeze. Default OFF => the json branches
        # of _pause_sitl/_unpause_sitl stay the #309 P1 no-op (byte-identical,
        # native path untouched either way). ON => SITL and its JSON-lockstep
        # FDM are SIGSTOP'd *together* across the SubprocVecEnv idle gaps that
        # T-29 MEASURED as ~460 s of uncommanded free-flight per sibling reset.
        self._json_pair_freeze = os.environ.get(
            "FD_JSON_PAIR_FREEZE", "0").strip() == "1"
        if self._json_pair_freeze and self._backend == "json":
            # Fail CLOSED. A finite FDM recv timeout keeps running while the
            # process is stopped (monotonic deadline) and fires the instant
            # SIGCONT lands — MEASURED, T-29 F-0 probe 2026-07-22: a 5 s
            # timeout under a 12 s freeze raised socket.timeout on resume even
            # though a packet was waiting. plane6dof_fdm.py:303 handles that by
            # BREAKING its serve loop => dead FDM on the first resume. The
            # blocking-socket setting (0 => `if self.recv_timeout_s:` falsy) is
            # the only safe pairing; liveness still covered by pdeathsig +
            # _kill_procs reaping.
            _rto = os.environ.get("FD_FDM_RECV_TIMEOUT_S", "30").strip()
            if _rto != "0":
                raise RuntimeError(
                    "FD_JSON_PAIR_FREEZE=1 requires FD_FDM_RECV_TIMEOUT_S=0 "
                    f"(got {_rto!r}). A finite FDM recv timeout expires during "
                    "the freeze and kills the FDM on resume (T-29 F-0 probe).")

        # R1 rider (2026-07-22): FD_INJECT_CONST_ACTION="a,b[,c]" — a FIXED
        # residual vector (normalized action units) substituted for the policy
        # action at the INJECT_ZERO_ACTION site. Unset/empty => None => inert.
        # Parse FAILS LOUD (a silent typo would fly arm-R as a no-op and the
        # authority witness would falsely HALT); values clipped to [-1, 1].
        _ica = os.environ.get("FD_INJECT_CONST_ACTION", "").strip()
        if _ica:
            try:
                _vec = np.array([float(x) for x in _ica.split(",")],
                                dtype=np.float32)
            except ValueError as e:
                raise RuntimeError(
                    f"FD_INJECT_CONST_ACTION unparseable: {_ica!r}") from e
            if _vec.shape != (config.ACTION_DIM,):
                raise RuntimeError(
                    f"FD_INJECT_CONST_ACTION needs {config.ACTION_DIM} comma-"
                    f"separated floats, got {_vec.shape[0]}: {_ica!r}")
            self._inject_const_action = np.clip(_vec, -1.0, 1.0)
            logger.warning("[R1] FD_INJECT_CONST_ACTION active: %s",
                           self._inject_const_action.tolist())
        else:
            self._inject_const_action = None

        # RESPID §3 micro-enabler (2026-07-24): probe-only CRRL bypass.
        # Default OFF => byte-identical. BANNED from campaign env lines
        # (same treatment as FD_ALIGN_MODE_CENSUS); response-ID legs only.
        self._respid_bypass_crrl = self._parse_respid_bypass_crrl()
        if self._respid_bypass_crrl:
            logger.warning("[RESPID] FD_RESPID_BYPASS_CRRL active: CRRL clip "
                           "bypassed (crrl_bounds=None); static clamps remain")

        # RESPID-2 enabler: phase-gated bounded-
        # terminal response-ID probe. Default OFF => byte-identical. BANNED
        # from campaign env lines; response-ID legs only. Single-axis STEP/
        # PULSE injected ONLY inside {DOCKING_CREEP, HOLD_CONTACT}; action == 0
        # (baseline-only) outside. Spec: analysis/crrl_ad_20260724T234726Z/.
        self._respid2_inject = self._parse_respid2_inject()
        self._respid2_win_tick = 0
        self._respid2_prev_in_win = False
        if self._respid2_inject is not None:
            logger.warning(
                "[RESPID2] FD_RESPID2_INJECT active: axis=%d wf=%s mag=%s "
                "(terminal-window only; action=0 outside window)",
                self._respid2_inject["axis"], self._respid2_inject["waveform"],
                self._respid2_inject["mag"])

        # T-25: training boot screen (default OFF => byte-identical).
        self._boot_screen_enabled = os.environ.get(
            "FD_TRAIN_BOOT_SCREEN", "0").strip() == "1"
        self._boot_screen_rerolls = 0        # rerolls consumed for the CURRENT external reset
        self._boot_screen_rerolling = False  # internal-recursion marker (never set externally)
        self._boot_screen_attempts_total = 0
        self._boot_screen_discards_total = 0
        self._boot_screen_max_discard_frac = float(
            os.environ.get("FD_B2_MAX_DISCARD_FRAC", "0.50") or "0.50")

        # P1-A item 9 (D1 rider): within-boot passive-vs-ticked align-mode
        # census. Default OFF => byte-identical. First boot per process only;
        # pure logging — it can decide T-33's D3-vs-D4 but changes NOTHING.
        self._align_census_enabled = os.environ.get(
            "FD_ALIGN_MODE_CENSUS", "0").strip() == "1"
        self._align_census_done = False   # process-lifetime latch
        self._align_census_state: Optional[Dict[str, Any]] = None  # armed dict

        # FIX-1 2026-07-05 (#320 RCA): episode-phase SIM_STATE read mode.
        # "1" (default) = passive cache read — consumes NO wire bytes on the
        # shared MAVLink connection; "0" = legacy unbounded recv_match drain
        # (rollback arm, preserved verbatim in _pump_mavlink).
        self._simstate_passive = os.environ.get(
            "FD_ENV_SIMSTATE_PASSIVE", "1").strip() == "1"

        # FIX-2 2026-07-05 (#319 RCA): json boot preposition hold time base.
        # "1" = sim-time holds — the preposition settle/arm-cadence/post-arm
        # dwells are measured from harvested follower time_boot_ms. #322
        # VERDICT: NOT adopted (default "0" = legacy wall holds). Measured:
        # the sim-time boot does NOT reduce M2-state entropy (behind_0 spread
        # 3.85 m ON vs 1.29 m OFF) and warmup_wall->m2_sim rho 0.988 is
        # near-tautological at speedup 1.0 (sim tracks wall under lockstep
        # pacing; m2_sim-warmup_wall constant to 0.073 s across 10 boots).
        # #323/#324 (2026-07-05): "1" is the STANDING ARM for ALL speedup-2
        # work — measured at 2x it keeps the boot sim-timeline speedup-
        # invariant (preposition ~48 sim-s at any speedup), which keeps
        # T_ENGAGE=76.0 inside the valid prep<bite<M2 window (the wall holds
        # at 2x span ~94 sim-s and would land the gate mid-preposition;
        # offw arm measured M2 ~+12 sim-s later). Default stays "0" for 1x
        # (#322 verdict unchanged); 2x drivers export =1 explicitly.
        self._simtime_warmup = os.environ.get(
            "FD_ENV_SIMTIME_WARMUP", "0").strip() == "1"

        # Gym spaces
        # D6.1 Amendment-O: the builder always emits 42-D (wind channels
        # deleted); wind_channels no longer selects the space dimension.
        obs_dim = config.OBS_DIM
        self.observation_space = spaces.Box(
            low=-config.OBS_CLIP,
            high=config.OBS_CLIP,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(config.ACTION_DIM,), dtype=np.float32,
        )

        # Subprocess handles (populated on reset)
        self._sitl_proc: Optional[subprocess.Popen] = None
        self._leader_proc: Optional[subprocess.Popen] = None
        self._fdm_proc: Optional[subprocess.Popen] = None  # #309 P1: JSON-lockstep FDM (json backend only)
        self._sitl_log_file = None    # SITL script log file handle
        self._leader_log_file = None  # Leader broadcaster log file handle
        self._fdm_log_file = None     # #309 P1: JSON FDM log file handle
        self._mav = None  # MAVLink connection (.mav from baseline MAVConnection)

        # Components (created on first reset)
        self._obs_builder: Optional[ObservationBuilder] = None
        self._action_proc: Optional[ResidualActionProcessor] = None
        self._reward_comp: Optional[MRRLRewardComputer] = None
        self._control_laws: Optional[FormationControlLaws] = None
        self._envelope: Optional[EnvelopeChecker] = None
        self._geom_calc: Optional[GeometryCalculator] = None
        self._time_auth: Optional[SimTimeAuthority] = None
        self._leader_rx = None  # LeaderTelemetryReceiver from warmup
        self._ctrl_ref = None  # DRLControllerRef (full FSM, Path 1)
        self._last_baseline_cmd: Optional[BaselineCommand] = None
        # T-438: r_safe wiring flag (default OFF => literal zeros, byte-
        # identical to the historical behavior). Elected mapping 2026-08-11:
        # breakaway guards ONLY (alt-ceiling + hard-overtake); speed-cap
        # excluded; safety_violations stays 0 (r_terminal already prices
        # terminal safety). Perception-side counters — G4-clean.
        self._rsafe_wire_enable = self._rsafe_wire_flag()
        # SHAPE-585: ceiling-shaping flag (default OFF => byte-identical; the
        # signed-amendment weights in config are inert until elected at a
        # launch seam). Endpoint form; prev latched at reset + each step end.
        self._ceiling_shape_enable = self._ceiling_shape_flag()
        self._ceiling_prev_below_m: Optional[float] = None
        # OVSHAPE-20260816: no-pass shaping flag (default OFF => byte-
        # identical; the signed-amendment weights in config are inert until
        # elected at a launch seam). Endpoint form, same seam as SHAPE-585.
        self._nopass_shape_enable = self._nopass_shape_flag()
        self._nopass_prev_behind_m: Optional[float] = None
        self._nopass_shape_max: float = float('-inf')
        self._nopass_shape_min: float = float('inf')
        # W-A (2026-07-29): formation-lost latch (armed on first ARM_STATES
        # sighting each episode; also reset in reset()). Validate the
        # configured state names against the real FSM enum ONCE, fail-closed.
        self._formation_lost_armed: bool = False
        if config.FORMATION_LOST_TERM_ENABLE:
            # ControllerState imported at module level (:77)
            _bad = [s for s in (config.FORMATION_LOST_STATES
                                + config.FORMATION_LOST_ARM_STATES)
                    if not hasattr(ControllerState, s)]
            if _bad:
                raise RuntimeError(
                    f"FD_FORMATION_LOST_TERM: unknown FSM states {_bad}")
        # RFL phase-gate hysteresis state (also reset in reset())
        self._rfl_active: bool = False
        self._rfl_activate_counter: int = 0
        self._rfl_deactivate_counter: int = 0
        # Bias EMA + extended logging (also reset in reset())
        self._bias_ema: float = 0.0
        self._voff_ema: float = 0.0  # VERTOFF-20260902: gated down-offset EMA, twin of _bias_ema
        self._c3o_policy_speed_delta_sum: float = 0.0
        self._c3o_policy_speed_delta_max: float = 0.0
        self._c3o_bias_speed_delta_sum: float = 0.0
        self._c3o_bias_active_ticks: int = 0
        self._c3o_bias_active_while_ahead_ticks: int = 0
        self._c3o_barrier_inside_ticks: int = 0
        self._c3o_barrier_outside_ticks: int = 0
        self._c3o_r_barrier_inside_sum: float = 0.0
        self._c3o_r_barrier_outside_sum: float = 0.0
        self._c3o_first_tier2_tick: int = -1
        self._c3o_first_bias_active_tick: int = -1
        self._c3o_ticks_to_90pct_bias: int = -1
        # Tier-4 hold-phase PSD instrumentation
        self._c3p_tier4_psd_values: list = []
        self._c3p_tier4_ticks: int = 0
        # Velocity modulation accumulator
        self._c3q_velocity_mod_values: list = []
        self._last_guided_alt_target: float = 0.0  # For GUIDED_CHANGE_ALTITUDE dedup
        self._last_guided_alt_time: float = 0.0
        self._last_reposition_time: float = 0.0
        self._last_reposition_hdg: float = 0.0
        self._last_speed_send_time: float = 0.0
        self._last_speed_send_value: float = 0.0
        self._last_reposition_alt: float = 0.0
        self._last_alt_cmd_target: float = 0.0
        self._alt_err_exceed_start: Optional[float] = None

        # Wake turbulence model (pure math, no SITL dependency)
        self._wake_model = (
            WakeTurbulenceZone(enable_wind_drift=True)
            if WakeTurbulenceZone is not None
            else None
        )

        # Home altitude for AGL safety check (None = not yet acquired from warmup)
        self._home_alt_msl: Optional[float] = None

        # Action scale (curriculum-dependent, updated in reset() and set_action_scale())
        self._action_scale: float = 1.0

        # Phase E Batch 3a: leader trajectory profile for current episode
        # (sampled in reset() from curriculum DR dict; default racetrack).
        self._current_leader_profile: str = "racetrack"

        # Phase E Batch 3c: per-episode DR sample passthrough for telemetry
        # (sampled in reset() from curriculum DR dict; defaults match S0 bounds).
        self._current_wind_speed_ms: float = 0.0
        self._current_wind_dir_deg: float = 0.0
        self._current_wind_turb: float = 0.0
        self._current_wind_sigma_ms: list = []   # T-393: real Dryden turbulence
        self._current_gps_accuracy: float = 0.3
        # 2026-07-29 v3 cell identity passthrough (the
        # curriculum cell dict carries these, but they were never stashed, so
        # metrics.csv v3_cell_label/v3_rehearsal were empty on every row).
        self._current_v3_cell_label: str = ""
        self._current_v3_rehearsal: str = ""
        # D2 (FD_CURRICULUM_V2): GPS-ladder state (rebuilt per-episode in reset;
        # inert in v1). Param family resolved once per FD_FW_VARIANT.
        self._curriculum_v2: bool = config.FD_CURRICULUM_V2
        self._gps_params = self._resolve_gps_param_names()
        self._gps_walk = None
        self._gps_dropout_sched = []
        self._gps_dropout_active = False
        self._gps_last_apply_sim_s = -1.0
        # Patch A 2026-05-06: stage-bound + harness flags for resolved telemetry.
        self._current_wind_max_ms: float = 0.0
        self._current_harness_uwb_enable: int = int(
            os.environ.get("HARNESS_UWB_ENABLE", "1")
        )
        self._current_reward_source_mode: str = os.environ.get(
            "FLIGHTDOCK_REWARD_SOURCE_MODE", "perception_only"
        )
        # Patch C2 2026-05-06: condition profile name (or None when env-var unset).
        self._current_condition_profile: Optional[str] = None
        # Patch C2 2026-05-06: SIM_STATE / RAW_TRUTH / paired-channel telemetry.
        self._sim_state_lat: Optional[float] = None
        self._sim_state_lon: Optional[float] = None
        self._sim_state_alt_m: Optional[float] = None
        self._sim_state_last_t: Optional[float] = None
        # FIX-1: parse timestamp of the last cache-observed SIM_STATE
        # (passive mode newness detection; None = nothing observed yet).
        self._sim_state_wire_stamp: Optional[float] = None
        # #337 Contract A: sender-side sim sample time (adjacent-ATTITUDE
        # tbms) of the cached SIM_STATE; -1 = no adjacent ATTITUDE (held).
        self._sim_state_att_tbms: int = -1
        # FIXA rolling lattice-center window (flag-gated consumer)
        self._align_center_win: deque = deque(maxlen=_RAW_ALIGN_CENTER_WIN)
        # #442: TRUE SIM_STATE wire velocities (NOT the GPI/EKF estimate) +
        # per-row raw-operand provenance snapshot (paired-CSV rm_* columns,
        # writer-only; offline de-ZOH instrument input, #443/#444).
        self._sim_state_vn: float = float("nan")
        self._sim_state_ve: float = float("nan")
        self._sim_state_vd: float = float("nan")
        self._raw_truth_meta: Optional[dict] = None
        self._raw_align2_held: int = 0
        self._raw_align2_used: int = 0
        # F-RAW (T-92) leg-level epoch-compensation state (flag-gated,
        # observability-only): 0 measuring, 1 latched-none, 2 latched-comp,
        # -1 fail-closed (marked, uncompensated).
        self._repoch_deltas: list = []
        self._repoch_state: int = 0
        self._repoch_ms: float = 0.0
        self._repoch_pairs: int = 0        # valid pairs seen (window cursor)
        self._repoch_win_a = None          # window-A estimate tuple
        self._repoch_guard_trips: int = 0  # consecutive off-latch windows
        # #475 R2 plausibility-hold state (write-time pairing-consistency
        # gate on the raw truth mirror; provenance mode 3; released after 4
        # consecutive so sustained real anomalies are never masked).
        self._raw_plaus_prev_b = None
        self._raw_plaus_prev_t = None
        self._raw_plaus_consec: int = 0
        self._raw_plaus_held: int = 0
        # #475 R3: follower-side pairing acceptance state (consume seam).
        self._r3_pair_rejects: int = 0
        self._r3_consec_rejects: int = 0
        self._r3_relocks: int = 0
        # #475 R4: kinematic stamp-synthesis state (consume seam); the bool
        # tracks whether the CACHED sample's stamp is synthetic (mode 5).
        self._r4_synth_accepts: int = 0
        self._r4_synth_declines: int = 0
        self._r5_restamps: int = 0   # #475 R5: lying present-stamps corrected
        self._sim_state_att_synth: bool = False
        # #343 v2.1: provenance of the CURRENT _raw_truth_geom (paired-row
        # column raw_align_mode): 0=fresh sender-aligned, 1=held reuse,
        # 2=unaligned passthrough, 3=#475 R2 plausibility hold, 4=#475 R3
        # follower pair-reject hold, 5=#475 R4 fresh-aligned on synthetic
        # stamp, -1=no raw geom.
        self._raw_align_mode: int = -1
        self._raw_truth_geom: Optional[RelativeGeometry] = None
        self._paired_csv_path: Optional[Path] = None
        self._paired_csv_file = None
        self._paired_csv_writer = None
        self._paired_episode_idx: int = 0
        self._fs_rec = None            # TELEM-766-A flight-state record
        # Per-episode rolling deltas for callback aggregates.
        self._paired_delta_env_raw_behind: list = []
        self._paired_delta_env_raw_cross: list = []
        self._paired_delta_env_raw_below: list = []
        self._paired_delta_ctrl_raw_behind: list = []
        self._paired_delta_ctrl_raw_cross: list = []
        self._paired_delta_ctrl_raw_below: list = []
        self._paired_hallucinated_env_m3b_count: int = 0
        self._paired_hallucinated_ctrl_m3b_count: int = 0
        # #452 belly-clamp: last SENT altitude command (tracked every tick so
        # docking-phase entry introduces no artificial step). None ⇒ no clamp.
        self._belly_prev_alt_cmd: Optional[float] = None
        # #461 fault-injection: accumulated upward alt-offset (0.0 = inert).
        self._faultinj_alt_offset: float = 0.0
        # Option-A1 cache-age telemetry (2026-05-09): track when ctrl_geom cache
        # last refreshed so triage can exclude stale-cache rows from RAW-CTRL
        # gate. Stale rows are preserved in CSV but marked.
        self._paired_total_row_count: int = 0
        self._ctrl_cache_last_behind_m: Optional[float] = None
        self._ctrl_cache_last_row_idx: int = 0
        self._ctrl_cache_last_wall_t: Optional[float] = None
        self._paired_uwb_terminal_accept: int = 0
        self._paired_uwb_terminal_reject: int = 0

        # Dispatch P — T0-SL runtime taint probe counters (FD_T0SL_PROBE=1 only;
        # default-OFF => byte-identical, counters inert; hot path gates on the
        # module-global _FD_T0SL_PROBE, not on any instance attr).
        self._t0sl_steps: int = 0
        self._t0sl_leaks: int = 0
        self._t0sl_max_obs: float = 0.0
        self._t0sl_max_reward: float = 0.0

        # Stage-dependent safety range (updated by set_safety_range)
        self._safety_range_m: float = config.STAGE_SAFETY_RANGE.get(0, config.HARD_RANGE_MAX_M)
        self._curriculum_stage: int = 0

        # v12: Wall-clock stall detector — catches SITL crash → frozen sim-time
        self._stall_detector = WallClockStallDetector(threshold_s=config.STALL_DETECTOR_THRESHOLD_S)
        self._stall_count: int = 0
        # C006 Phase 7: Action accumulator for per-episode residual stats
        self._action_accum: list = []

        # v13 Commit 12: Per-worker heartbeat path
        self._heartbeat_path = Path(
            config.HEARTBEAT_PER_WORKER_TEMPLATE.format(env_id=self.env_id)
        )

        # v18 Phase C: Warmup completion flag — asserted in step() to prevent
        # warmup transitions from entering the replay buffer.
        self._warmup_complete: bool = False

        # Episode state
        self._episode_start_sim_s: float = 0.0
        self._tick_count: int = 0
        # TERMDEMOTE-767: terminal -> one raw-fact event record. Declared
        # here AND re-bound in reset(), matching _tick_count above, so a
        # pre-reset _check_termination can never hit an AttributeError.
        self._demoted_events: Dict[str, dict] = {}
        # NOPASSTERM-770: consecutive-tick counter for SAFETY_NOPASS, declared
        # here AND re-bound in reset() for the same pre-reset reason.
        self._nopass_ticks: int = 0
        # TELEMFIX-771 per-channel arrival tracking and abstention counts.
        self._telem_seen: Dict[str, object] = {}
        self._telem_age_s: Dict[str, float] = {}
        self._telem_abstains: Dict[str, int] = {}
        self._continuous_m3b_dwell_s: float = 0.0
        self._accumulated_m3b_dwell_s: float = 0.0
        self._milestone_tier: int = 0

        # v19: True FlightDock contact milestones (boom-tip reference, diagnostic only)
        self._continuous_contact_m3a_dwell_s: float = 0.0
        self._contact_m3a_achieved: bool = False
        self._continuous_contact_m3b_dwell_s: float = 0.0
        self._accumulated_contact_m3b_dwell_s: float = 0.0
        self._best_contact_m3b_dwell_s: float = 0.0
        self._boom_tip_min_range_m: float = float('inf')
        # C006 P4: Actual milestone tracking (M2 + M3a coarse, boom-tip reference)
        self._continuous_actual_m2_dwell_s: float = 0.0
        self._best_actual_m2_dwell_s: float = 0.0
        self._continuous_contact_m3a_coarse_dwell_s: float = 0.0
        self._contact_m3a_coarse_achieved: bool = False
        # Campaign 3c: dwell-failure decomposition (diagnostic)
        self._m3b_entry_count: int = 0
        self._m3b_break_count: int = 0
        self._m3b_total_dwell_before_break_s: float = 0.0
        self._m3b_last_break_axis: str = "none"
        self._m3b_last_break_margin_m: float = -999.0   # sentinel: no break yet
        self._m3b_last_break_velocity_mps: float = -1.0  # sentinel: no break yet
        self._m3b_break_axis_counts: Dict[str, int] = {"behind": 0, "cross": 0, "below": 0}
        self._m3b_was_in_contact: bool = False
        # Campaign 3h: exit penalty diagnostics
        self._penalized_exit_count: int = 0
        self._pre_exit_dwell_accum: float = 0.0
        self._pre_exit_dwell_count: int = 0
        # reward pivot diagnostics
        self._approach_suppression_active: bool = False
        self._r_approach_suppressed_ticks: int = 0
        # R0 (REWARDHYG-698): keys the reward path ACTUALLY produces this
        # lifetime. The trimmed-term declaration is DERIVED from this set
        # minus _MRRL_RAW_COMPONENTS, never hand-authored -- so it can
        # disagree if a term is un-trimmed without the list being edited.
        self._mrrl_keys_seen: set = set()
        self._behind_axis_exit_count: int = 0
        self._non_behind_exit_count: int = 0
        self._r_exit_behind_total: float = 0.0
        self._ttc_at_m3b_entries: list = []
        self._closure_rate_delta_in_m3b: list = []
        # Overspeed penalty diagnostics
        self._r_overspeed_total: float = 0.0
        self._overspeed_tick_count: int = 0
        self._max_v_excess: float = 0.0
        self._tier_first_achieved_s: Dict[int, float] = {}
        self._tier_dwell_s: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self._min_range_m: float = float('inf')
        self._max_behind_overshoot_m: float = 0.0
        self._max_cross_excursion_m: float = 0.0
        # T-455: per-episode vertical + guard
        # instrumentation, TELEMETRY ONLY. These are written to the metrics.csv
        # row and read by NOTHING else -- not the observation, not the reward,
        # not the replay buffer, not any policy update. G4 Tier-0 containment is
        # asserted mechanically by tests/unit/drl/test_t455_vertical_instrumentation.py,
        # which parses this file and requires each truth-side name to appear at
        # exactly four sites (init / reset / update / emit) and nowhere else.
        #
        # Sign convention matches the scorer: below_m is NEGATIVE above the boom
        # tip, so peak-up = max(0, -below_m) (cf. gate_safety_events.py:213).
        #
        # WARNING -- the truth channel is a 10 Hz HOLD (~49% of ticks repeat the
        # previous sample; T-441/443-446). Extrema over held samples are valid.
        # A PER-TICK RATE COMPUTED FROM THESE COLUMNS IS NOT. Do not difference
        # them.
        self._max_above_boom_fused_m: float = 0.0
        self._max_above_boom_truth_m: float = 0.0
        self._min_boom_tip_range_truth_m: float = float('inf')
        self._guard_fires_episode: int = 0
        # Baseline for the guard-fire DELTA. Latched lazily on the first tick
        # that can see the controller, because _ctrl_ref is not guaranteed to
        # exist at reset(). None => not yet latched.
        self._guard_fires_base: Optional[int] = None
        self._prev_applied_residual = np.zeros(3, dtype=np.float32)
        self._prev_range_m: float = 0.0

        # Idle-drift fix (v7): SIGSTOP/SIGCONT pause tracking
        self._pause_count: int = 0
        self._unpause_count: int = 0
        self._max_pause_wall_s: float = 0.0
        self._pause_wall_start: Optional[float] = None
        # v11: sim-time correlation for definitive idle-drift verification
        self._total_stopped_wall_s: float = 0.0
        self._last_sitl_sim_at_stop: float = 0.0
        self._last_sitl_sim_at_resume: float = 0.0

        # Telemetry cache
        self._follower_state: Optional[FollowerState] = None
        self._leader_state: Optional[LeaderState] = None
        self._last_geom: Optional[RelativeGeometry] = None
        # T-71 ENV ALIGN2 provenance (2 = unaligned passthrough, 0 = aligned).
        self._env_align_mode: int = 2
        self._env_align_age_s: float = 0.0
        # Position and velocity cache for geometry computation
        self._follower_lat: float = -35.3632622
        self._follower_lon: float = 149.1652374
        self._follower_vn: float = 0.0
        self._follower_ve: float = 0.0
        self._leader_lat: float = -35.3632622
        self._leader_lon: float = 149.1652374
        self._leader_vn: float = 0.0
        self._leader_ve: float = 0.0

        # Gated leader-follower diagnostic (Option C, SMOKE-A2-B2):
        # gate is read once at construction so the per-tick hot path is a
        # single bool check.
        self._lead_foll_diag_enabled: bool = (
            os.environ.get("FLIGHTDOCK_LEADFOLL_DIAG", "0") == "1"
        )

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def _realization_seeds(self, ep_idx: int, attempt: int = 1):
        """WARMUPREROLL: the two REALIZATION
        words -- (leader weather seed, FDM gust seed) -- for ONE warmup attempt.

        attempt 1: exactly the pre-fix lineage, SeedSequence([base, env_id,
        ep_idx]) words [4], [5] (seed-v2 OFF: weather None + static
        FD_FDM_WIND_SEED mix) -- byte-identical, so every banked corpus and
        the (seed, env_id, ep_idx) pairing label are untouched.
        attempt >= 2 with FD_WARMUP_REROLL != "0": SeedSequence([base, env_id,
        ep_idx, attempt]) -- a lineage disjoint from every 3-tuple and from
        the GPS walk's [.., 0x6753]. FD_WARMUP_REROLL=0 => attempt 1's words
        on every attempt (the pre-fix replay; kill-switch). DR-rng lineage,
        never truth (G4); the cell (words [0:4]) is NOT re-rolled here.
        """
        reroll = (attempt >= 2
                  and os.environ.get("FD_WARMUP_REROLL", "1") != "0")
        if self._dr_seed_v2 and self._initial_seed is not None:
            ent = [int(self._initial_seed), int(self.env_id), int(ep_idx)]
            if reroll:
                ent.append(int(attempt))
            w = np.random.SeedSequence(ent).generate_state(8)
            return int(w[4]), int(w[5])
        base = int(os.environ.get("FD_FDM_WIND_SEED", "20260622") or "20260622")
        mix = 104729 * (int(attempt) - 1) if reroll else 0
        return None, ((base + 1000003 * int(self.env_id)
                       + 8191 * int(ep_idx) + mix) & 0x7FFFFFFF)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Kill old SITL, launch fresh, warmup to M2, return first DRL obs."""
        # Rev16 §3.1(a) — clear stale state FIRST.
        self._current_ep_dir = None
        self._current_ep_idx = None

        # Rev16 §3.1(b) — snapshot-then-increment AT START.
        ep_idx = self._episode_index
        self._episode_index += 1

        # Rev16 §3.1(c) — EP_DIR compute (only if out_dir configured).
        if self._out_dir is not None:
            ep_dir = os.path.join(
                self._out_dir,
                f"episode_{ep_idx:03d}_env{self.env_id}",
            )
            os.makedirs(ep_dir, exist_ok=True)
        else:
            ep_dir = None

        # Rev16 §3.1(d) — staging assigns (consumed by _run_warmup).
        self._current_ep_dir = ep_dir
        self._current_ep_idx = ep_idx

        # Phase G: if caller omits seed AND a constructor-time _initial_seed
        # was supplied AND this is the first reset, use the stored seed.
        # Known behavioral boundary: the _first_reset_done gate fires once per
        # env object lifetime, not once per training run. Acceptable for C4
        # (workers not recycled across runs). See PHASE_G_IMPLEMENTATION_REPORT.
        if seed is None and self._initial_seed is not None and not self._first_reset_done:
            seed = self._initial_seed
        self._first_reset_done = True
        super().reset(seed=seed)

        # v18 Phase C: Mark warmup incomplete — step() asserts this is True.
        self._warmup_complete = False

        # v13-fix: Touch per-worker heartbeat at reset() entry to bridge the
        # training-update → warmup gap. During on-policy training updates (PPO),
        # step() doesn't run so the per-worker heartbeat file goes stale.
        # Without this touch, the watchdog can false-positive kill the process
        # if the training update + warmup exceeds the stale threshold.
        if hasattr(self, '_heartbeat_path'):
            self._heartbeat_path.touch(exist_ok=True)

        # Kill any existing processes
        self._kill_procs()

        # v19 Fix 5: Refresh heartbeat after potentially slow process cleanup.
        # _kill_procs() can take 5-10s (SIGTERM→wait→SIGKILL per proc).
        # Without this, heartbeat age grows before warmup even starts.
        if hasattr(self, '_heartbeat_path'):
            self._heartbeat_path.touch(exist_ok=True)

        # Domain randomisation
        wind_spd = self.wind_speed_ms
        wind_dir = 0.0
        wind_turb = self.wind_turb
        gps_accuracy = 0.3  # ArduPilot default
        gps_dropout_enabled = False  # D2: GPS dropout windows (v2 S4a+ only)
        gps_dropout_max = 0
        warmup_target = "M2"  # default
        action_scale = 1.0  # default (full authority)
        # Phase E Batch 3a: per-episode leader trajectory profile (Alternative B mix)
        leader_profile = "racetrack"  # default when curriculum unavailable
        leader_omega = None
        wind_sigma_ms = []            # T-393: the REAL (Dryden) turbulence
        # D1 (FD_DR_SEED_V2): build the per-episode deterministic RNG + leader
        # weather seed from ONE SeedSequence([base, env_id, ep_idx]); hand the
        # Generator to the curriculum for DR draws. Flag OFF => None on both
        # (legacy global-RNG DR + no --weather-seed) => byte-identical.
        if self._dr_seed_v2 and self._initial_seed is not None:
            _ss = np.random.SeedSequence(
                [int(self._initial_seed), int(self.env_id), int(ep_idx)]
            )
            self._episode_rng = np.random.default_rng(_ss.generate_state(4))
        else:
            self._episode_rng = None
        # #465 word [4] (leader weather) + word [5] (FDM gust) from the SAME
        # lineage as the DR rng words [0:4]; seed-v2 OFF => weather None and
        # the static FD_FDM_WIND_SEED mix. WARMUPREROLL (2026-08-29): both are
        # the REALIZATION words, re-derived per warmup attempt inside the
        # retry loop below; attempt 1 here is byte-identical to the pre-fix
        # derivation (DR-rng lineage, never truth; G4).
        self._episode_weather_seed, self._episode_wind_seed = (
            self._realization_seeds(ep_idx, 1))
        if self.curriculum_manager is not None:
            self.curriculum_manager._episode_rng = self._episode_rng
            dr = self.curriculum_manager.sample_domain_randomization()
            wind_spd = dr.get("wind_speed_ms", wind_spd)
            wind_dir = dr.get("wind_dir_deg", wind_dir)
            wind_turb = dr.get("wind_turb", wind_turb)
            gps_accuracy = dr.get("gps_accuracy", gps_accuracy)
            gps_dropout_enabled = dr.get("gps_dropout_enabled", gps_dropout_enabled)
            gps_dropout_max = dr.get("gps_dropout_max_per_episode", gps_dropout_max)
            warmup_target = dr.get("warmup_target", "M2")
            action_scale = dr.get("action_scale", 1.0)
            leader_profile = dr.get("leader_profile", leader_profile)
            leader_omega = dr.get("leader_omega_deg_s", leader_omega)
            wind_sigma_ms = list(dr.get("wind_sigma_ms", []) or [])
            # 2026-07-29: v3 cell identity (empty-string on non-v3 cells so
            # the metrics column stays additive; consumer is callbacks.py
            # info.get passthrough — no training-side reader).
            self._current_v3_cell_label = str(dr.get("v3_cell_label", ""))
            self._current_v3_rehearsal = str(dr.get("v3_rehearsal", ""))
        # Patch C2 2026-05-06: FLIGHTDOCK_CONDITION_PROFILE env-var fallback.
        # Hard-fail on unknown profile name.
        # Resolution order: CONDITION_PROFILES override -> curriculum DR -> defaults.
        condition_profile_name = os.environ.get("FLIGHTDOCK_CONDITION_PROFILE")
        if condition_profile_name is not None:
            if condition_profile_name not in config.CONDITION_PROFILES:
                valid = sorted(config.CONDITION_PROFILES.keys())
                raise RuntimeError(
                    f"[env_id={self.env_id}] FLIGHTDOCK_CONDITION_PROFILE="
                    f"{condition_profile_name!r} is not a valid profile. "
                    f"Valid options: {valid}"
                )
            profile = config.CONDITION_PROFILES[condition_profile_name]
            # 'curriculum' sentinel: keep curriculum-sampled values; only annotate.
            if condition_profile_name != "curriculum":
                if profile["wind_max_ms"] is not None:
                    wind_spd = float(profile["wind_max_ms"])
                if profile["wind_turb"] is not None:
                    wind_turb = float(profile["wind_turb"])
                if profile["gps_accuracy"] is not None:
                    gps_accuracy = float(profile["gps_accuracy"])
                if profile["leader_profile"] is not None:
                    leader_profile = str(profile["leader_profile"])
                if profile["leader_omega_deg_s"] is not None:
                    leader_omega = float(profile["leader_omega_deg_s"])
            self._current_condition_profile = condition_profile_name
            logger.info(
                f"[env_id={self.env_id}] [FLIGHTDOCK_CONDITION_PROFILE] "
                f"resolved: {condition_profile_name!r} -> wind_spd={wind_spd}, "
                f"wind_turb={wind_turb}, gps_accuracy={gps_accuracy}, "
                f"leader_profile={leader_profile!r}"
            )
        # T-229: OFF-LADDER wind override for the
        # H2 saturation probe. The ladder tops at 10 m/s; probing 11-15 m/s
        # needs a wind the curriculum cannot resolve. Absent => byte-identical
        # behavior (eval-only escape, never set by any training driver).
        # Clamped to [0, 15] per the elected probe ladder. Applies to the
        # LEADER + episode config here and to the FDM at its own env read —
        # both sites, one flag, so the two aircraft always share one air mass.
        _wov = os.environ.get("FD_EVAL_WIND_OVERRIDE_MS", "").strip()
        if _wov:
            wind_spd = min(max(float(_wov), 0.0), 15.0)
            logger.info(
                f"[env_id={self.env_id}] [WIND_OVERRIDE] "
                f"FD_EVAL_WIND_OVERRIDE_MS={_wov} -> wind_spd={wind_spd} "
                f"(off-ladder probe cell; leader+FDM share this wind)")
        else:
            self._current_condition_profile = None

        # Route α: HARNESS_LEADER_PROFILE override for Method-B / Method-C symmetry.
        env_profile = os.environ.get("HARNESS_LEADER_PROFILE")
        if env_profile:
            logger.info(
                f"[env_id={self.env_id}] [HARNESS_LEADER_PROFILE] override: "
                f"{leader_profile!r} -> {env_profile!r}"
            )
            leader_profile = env_profile
        self._action_scale = action_scale
        self._current_leader_profile = leader_profile  # per-episode telemetry
        # Phase E Batch 3c: stash DR sample on self for info-dict telemetry passthrough
        self._current_wind_speed_ms = wind_spd
        self._current_wind_dir_deg = wind_dir
        self._current_wind_turb = wind_turb
        # T-393: the REAL turbulence. `wind_turb` above is the legacy
        # SIM_WIND_TURB knob, which is INERT on the json backend (curriculum.py
        # :412-414) -- it reads 0.0 on cells that are genuinely turbulent, so
        # anything that reports it alone is actively misleading. The flown
        # turbulence is the FDM Dryden sigma triple.
        self._current_wind_sigma_ms = wind_sigma_ms
        self._current_gps_accuracy = gps_accuracy
        # Patch A 2026-05-06: stage-bound and harness flags for resolved telemetry.
        try:
            self._current_wind_max_ms = float(
                config.CURRICULUM_STAGES[self._curriculum_stage].wind_max_ms
            )
        except (KeyError, AttributeError):
            self._current_wind_max_ms = 0.0
        self._current_harness_uwb_enable = int(
            os.environ.get("HARNESS_UWB_ENABLE", "1")
        )
        self._current_reward_source_mode = os.environ.get(
            "FLIGHTDOCK_REWARD_SOURCE_MODE", "perception_only"
        )

        # Launch SITL + leader + warmup with retry resilience.
        # Transient failures (SITL startup timeout, port-in-use, MAVLink
        # connection drop) are retried up to MAX_WARMUP_RETRIES times.
        # Programming bugs (AttributeError, TypeError) propagate immediately.
        last_error: Optional[Exception] = None
        for attempt in range(1, config.MAX_WARMUP_RETRIES + 1):
            # WARMUPREROLL: a refused/timed-out attempt gets a FRESH gust +
            # leader-weather realization (attempt 1 == pre-fix lineage; the
            # cell -- DR rng, wind_spd/dir/turb -- is never re-rolled).
            self._warmup_attempt = attempt
            self._episode_weather_seed, self._episode_wind_seed = (
                self._realization_seeds(ep_idx, attempt))
            try:
                t0 = time.time()
                # #309 P1: json backend spawns the external FDM FIRST so it is
                # bound on self._fdm_port before SITL (--model JSON) connects.
                # Fresh FDM per retry; _kill_procs() reaps the stale one.
                if self._backend == "json":
                    self._launch_fdm()
                self._launch_sitl()
                logger.info(
                    f"[env_id={self.env_id}] _launch_sitl() returned in {time.time()-t0:.2f}s")
                sys.stdout.flush()  # v19 Fix 4: flush through tee pipe
                t1 = time.time()
                # FD_LEADER_WIND (2026-07-27 wind-plumbing fix, default OFF):
                # "1" passes the resolved cell wind to the leader so both
                # aircraft share one air mass (the leader's
                # sync_weather_to_sitl push of SIM_WIND_* remains inert on the
                # JSON follower — its wind is the external FDM's). Unset/0 =
                # legacy split atmosphere: leader launched windless (hardcoded
                # zeros since c69a6789; the old comment's "DR wind applied
                # post-warmup via SIM_WIND_* params" never reached the leader).
                # Trajectory profile IS set at launch time — leader runs for
                # the whole episode with this profile.
                if os.environ.get("FD_LEADER_WIND", "0") == "1":
                    _lw_spd, _lw_dir, _lw_turb = wind_spd, wind_dir, wind_turb
                else:
                    _lw_spd, _lw_dir, _lw_turb = 0.0, 0.0, 0.0
                self._launch_leader(
                    _lw_spd, _lw_dir, _lw_turb,
                    profile=leader_profile,
                    omega_deg_s=leader_omega,
                    weather_seed=self._episode_weather_seed,
                )
                logger.info(
                    f"[env_id={self.env_id}] _launch_leader() returned in {time.time()-t1:.2f}s")
                sys.stdout.flush()  # v19 Fix 4: flush through tee pipe

                # === Phase 5c.A.5 HARNESS EQUALIZATION ===
                # Bring eval/training path to gated_matrix preflight parity.
                # Window: leader is up; env's pymavlink master has NOT yet
                # bound udp:14551+offset (binding happens in _run_warmup at
                # DRLControllerRef construction). Subprocess preflights
                # claim/release that port without EADDRINUSE.
                if os.environ.get("FLIGHTDOCK_HARNESS_EQUALIZATION", "0") == "1":
                    from .harness_equalization import run_preflight
                    t_eq = time.time()
                    run_preflight(
                        follower_port=self._follower_port,
                        speedup=self.speedup,
                        out_dir=self._current_ep_dir or self._out_dir,
                        ep_idx=self._current_ep_idx,
                        env_id=self.env_id,
                    )
                    logger.info(
                        f"[env_id={self.env_id}] harness_equalization preflight "
                        f"completed in {time.time()-t_eq:.1f}s")
                    sys.stdout.flush()

                # #309 P1: census-substrate preposition (arm+GUIDED+cruise)
                # BEFORE the skip_takeoff warmup — json only, native unchanged.
                if self._backend == "json":
                    t_pp = time.time()
                    self._preposition_airborne()
                    logger.info(
                        f"[env_id={self.env_id}] _preposition_airborne() "
                        f"returned in {time.time()-t_pp:.1f}s")
                    sys.stdout.flush()

                t2 = time.time()
                self._run_warmup(warmup_target=warmup_target)
                logger.info(
                    f"[env_id={self.env_id}] _run_warmup() completed in {time.time()-t2:.1f}s")
                sys.stdout.flush()  # v19 Fix 4: flush through tee pipe
                # B+ Step B trigger (no-op when env var unset).
                self._request_controller_fusion_reset()
                last_error = None
                break
            except (TimeoutError, ConnectionError, OSError) as e:
                last_error = e
                tag = f"[DRL env_id={self.env_id}]"
                logger.warning(f"{tag} Warmup attempt {attempt}/{config.MAX_WARMUP_RETRIES} "
                               f"failed after {time.time()-t0:.1f}s: {type(e).__name__}: {e}")
                sys.stdout.flush()  # v19 Fix 4: flush before retry sleep
                self._kill_procs()
                if attempt < config.MAX_WARMUP_RETRIES:
                    time.sleep(config.WARMUP_RETRY_DELAY_S)

        if last_error is not None:
            raise RuntimeError(
                f"Warmup failed after {config.MAX_WARMUP_RETRIES} attempts. "
                f"Last error: {last_error}"
            ) from last_error

        # === T-25: boot screen (flag-gated; default OFF => this block is inert).
        # Placement is load-bearing: BEFORE _apply_gps_degradation (clean-GPS
        # sample; m_a is datalink-side and GPS-independent), BEFORE
        # _open_paired_csv (a rerolled boot leaves NO artifact, so positional
        # pairing of paired CSVs with metrics rows cannot be disturbed), and
        # BEFORE the first step() (no transition exists yet => a fired boot can
        # never reach the replay buffer). Reroll = _kill_procs + recursive
        # re-entry, bounded by _BOOT_SCREEN_MAX_REROLLS; cap-exhausted boots
        # are KEPT and flagged (a training run must not deadlock).
        if self._boot_screen_enabled:
            if not self._boot_screen_rerolling:
                self._boot_screen_rerolls = 0
            self._boot_screen_rerolling = False
            _bs = self._boot_screen_classify()
            self._boot_screen_attempts_total += 1
            if (_bs["fires"]
                    and self._boot_screen_rerolls >= _BOOT_SCREEN_MAX_REROLLS):
                _bs["action"] = "keep_cap_exhausted"
            elif _bs["fires"]:
                _bs["action"] = "discard"
            elif not _bs["usable"]:
                _bs["action"] = "keep_fail_open"
            else:
                _bs["action"] = "keep"
            self._boot_screen_record(_bs)
            if _bs["action"] == "discard":
                self._boot_screen_discards_total += 1
                a_tot = self._boot_screen_attempts_total
                d_tot = self._boot_screen_discards_total
                if (a_tot >= _BOOT_SCREEN_HALT_MIN_ATTEMPTS
                        and d_tot / a_tot > self._boot_screen_max_discard_frac):
                    raise RuntimeError(
                        f"[BOOT_SCREEN] HALT: discard fraction {d_tot}/{a_tot} "
                        f"exceeds {self._boot_screen_max_discard_frac} — rig "
                        "fault, not lottery (T-26 R3 / PREREG_513 machinery).")
                self._boot_screen_rerolls += 1
                logger.warning(
                    "[BOOT_SCREEN] env_id=%s FIRED (age2_med=%.1f ms, "
                    "bias_med=%.3f m) — discarding boot, reroll %d/%d",
                    self.env_id, _bs["age2_med_ms"] or float("nan"),
                    _bs["bias_med_m"] or float("nan"),
                    self._boot_screen_rerolls, _BOOT_SCREEN_MAX_REROLLS)
                self._kill_procs()
                self._boot_screen_rerolling = True
                return self.reset(seed=seed, options=options)

        # P1-A item 9: within-boot align census. Placement mirrors the screen's
        # (clean GPS, before paired CSV, before first step) and runs on the
        # KEPT boot when the screen rerolls, preserving the within-boot
        # property. Default OFF => inert. First boot per process only.
        self._align_census_abort_if_armed()   # never let a window span boots
        self._align_census_passive()

        # Set stage-dependent GPS degradation on follower SITL (AFTER warmup so
        # warmup uses clean GPS). v2 builds the seeded OU GLTCH walk + dropout;
        # v1 keeps the legacy reported-accuracy write (byte-identical).
        self._apply_gps_degradation(gps_accuracy, gps_dropout_enabled, gps_dropout_max)

        # Apply DR wind AFTER warmup (launched with wind=0 for reliable warmup)
        self._set_sim_param("SIM_WIND_SPD", wind_spd)
        self._set_sim_param("SIM_WIND_DIR", wind_dir)
        self._set_sim_param("SIM_WIND_TURB", wind_turb)

        # Create / reset DRL-specific components (obs builder, action proc, etc.)
        self._init_components(stage=self._curriculum_stage)

        # Seed geometry from warmup handoff state so first _build_observation()
        # sees a populated _last_geom (not None → avoids all-zero first obs).
        self._compute_geometry()

        # T-434 witness (FM7): announce the active collision-judge channel so
        # a leg log proves the flag was live, not merely exported.
        if config.COLLISION_TRUTH_CHANNEL_ENABLE:
            logger.info("[T434] SAFETY_COLLISION judge channel: TRUTH "
                        "(FD_COLLISION_TRUTH_CHANNEL=1; per-tick GPS fallback)")
        # NOPASSTERM-770 witness: a leg log proves which terminals were live
        # and at what no-pass line, not merely that the knob was exported.
        if config.SAFETY_TERMINALS:
            logger.info("[NOPASSTERM-770] live terminals %s; no-pass line "
                        "%.2f m ahead of tip, %d ticks (%.2f s); contact "
                        "plane below_m < %.2f",
                        sorted(config.SAFETY_TERMINALS),
                        config.NOPASS_TERM_AHEAD_M,
                        config.NOPASS_TERM_PERSIST_TICKS,
                        config.NOPASS_TERM_PERSIST_S,
                        config.CONTACT_PLANE_BELOW_M)
            logger.info("[NOPASSGATE-781] no-pass off-axis gate %.2f m "
                        "(sqrt(cross^2+below^2)); set huge to disable",
                        config.NOPASS_TERM_OFFAXIS_M)

        # T-438 witness: announce the r_safe wiring state so a leg log
        # proves the flag was live, not merely exported.
        if self._rsafe_wire_enable:
            logger.info("[T438] r_safe wiring: ACTIVE "
                        "(FD_RSAFE_WIRE_ENABLE=1; guard_fires = per-sub-tick "
                        "breakaway-guard delta, speed-cap excluded)")

        # SHAPE-585 witness: announce the ceiling-shaping state so a leg log
        # proves the flag was live, not merely exported.
        if self._ceiling_shape_enable:
            logger.info("[SHAPE585] ceiling shaping: ACTIVE "
                        "(FD_CEILING_SHAPE_ENABLE=1; endpoint asym+PBRS, "
                        "W_ABOVE_ASYM=%.3f W_CEILING=%.3f ramp_start=%.2f m "
                        "gamma=%.3f)", config.W_ABOVE_ASYM, config.W_CEILING,
                        config.CEILING_RAMP_START_M, config.CEILING_SHAPE_GAMMA)

        # OVSHAPE witness: announce the no-pass shaping state so a leg log
        # proves the flag was live, not merely exported.
        if self._nopass_shape_enable:
            logger.info("[OVSHAPE] no-pass shaping: ACTIVE "
                        "(FD_NOPASS_SHAPE_ENABLE=1; endpoint asym+PBRS, "
                        "W_INFRONT_ASYM=%.3f W_NOPASS=%.3f ramp_start=%.2f m "
                        "cap=%.2f m gamma=%.3f)", config.W_INFRONT_ASYM,
                        config.W_NOPASS, config.NOPASS_RAMP_START_M,
                        config.NOPASS_CAP_M, config.NOPASS_SHAPE_GAMMA)

        # Record DRL episode start time
        self._episode_start_sim_s = self._time_auth.now_s()
        self._tick_count = 0
        self._stall_detector.reset()
        self._continuous_m3b_dwell_s = 0.0
        self._accumulated_m3b_dwell_s = 0.0
        self._continuous_m3a_fine_dwell_s = 0.0
        self._milestone_tier = 0
        # v19: Contact milestone resets
        self._continuous_contact_m3a_dwell_s = 0.0
        self._contact_m3a_achieved = False
        self._continuous_contact_m3b_dwell_s = 0.0
        self._accumulated_contact_m3b_dwell_s = 0.0
        self._best_contact_m3b_dwell_s = 0.0
        self._boom_tip_min_range_m = float('inf')
        # C006 P4: Actual milestone resets
        self._continuous_actual_m2_dwell_s = 0.0
        self._best_actual_m2_dwell_s = 0.0
        self._continuous_contact_m3a_coarse_dwell_s = 0.0
        self._contact_m3a_coarse_achieved = False
        # Campaign 3c: dwell-failure decomposition resets
        self._m3b_entry_count = 0
        self._m3b_break_count = 0
        self._m3b_total_dwell_before_break_s = 0.0
        self._m3b_last_break_axis = "none"
        self._m3b_last_break_margin_m = -999.0
        self._m3b_last_break_velocity_mps = -1.0
        self._m3b_break_axis_counts = {"behind": 0, "cross": 0, "below": 0}
        self._m3b_was_in_contact = False
        # Campaign 3h: exit penalty diagnostics
        self._penalized_exit_count = 0
        self._pre_exit_dwell_accum = 0.0
        self._pre_exit_dwell_count = 0
        # reward pivot diagnostics
        self._approach_suppression_active = False
        self._r_approach_suppressed_ticks = 0
        self._behind_axis_exit_count = 0
        self._non_behind_exit_count = 0
        self._r_exit_behind_total = 0.0
        self._ttc_at_m3b_entries = []
        self._closure_rate_delta_in_m3b = []
        # Overspeed penalty diagnostics
        self._r_overspeed_total = 0.0
        self._overspeed_tick_count = 0
        self._max_v_excess = 0.0
        self._tier_first_achieved_s = {}
        self._tier_dwell_s = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self._min_range_m = float('inf')
        self._max_behind_overshoot_m = 0.0
        self._max_cross_excursion_m = 0.0
        self._max_above_boom_fused_m = 0.0      # T-455
        self._max_above_boom_truth_m = 0.0
        self._min_boom_tip_range_truth_m = float('inf')
        self._guard_fires_episode = 0
        self._guard_fires_base = None

        # Patch C2 2026-05-06: paired-channel CSV reset + open new writer.
        self._paired_episode_idx += 1
        self._paired_delta_env_raw_behind = []
        self._paired_delta_env_raw_cross = []
        self._paired_delta_env_raw_below = []
        self._paired_delta_ctrl_raw_behind = []
        self._paired_delta_ctrl_raw_cross = []
        self._paired_delta_ctrl_raw_below = []
        self._paired_hallucinated_env_m3b_count = 0
        self._paired_hallucinated_ctrl_m3b_count = 0
        self._paired_total_row_count = 0
        self._ctrl_cache_last_behind_m = None
        self._ctrl_cache_last_row_idx = 0
        self._ctrl_cache_last_wall_t = None
        self._belly_prev_alt_cmd = None   # #452: clear per episode
        self._faultinj_alt_offset = 0.0   # #461: re-arm the injection ramp per episode
        self._paired_uwb_terminal_accept = 0
        self._paired_uwb_terminal_reject = 0
        self._raw_truth_geom = None
        self._sim_state_lat = None
        self._sim_state_lon = None
        self._sim_state_alt_m = None
        self._sim_state_last_t = None
        self._sim_state_wire_stamp = None
        self._sim_state_att_tbms = -1  # #337: sender-side stamp reset
        self._align_center_win.clear()  # FIXA: center re-learned per leg
        self._epoch_probe_count = 0    # T-165: probe budget resets per episode
        self._sim_state_vn = float("nan")     # #442: TRUE wire velocities reset
        self._sim_state_ve = float("nan")
        self._sim_state_vd = float("nan")
        self._raw_truth_meta = None           # #442: operand-provenance snapshot
        self._raw_align2_held = 0
        self._raw_align2_used = 0
        self._repoch_deltas = []   # F-RAW: per-episode re-estimate (T-92 boot
        self._repoch_state = 0     # latch is per-boot; per-episode reset is
        self._repoch_ms = 0.0      # strictly safer and self-verifying)
        self._repoch_pairs = 0
        self._repoch_win_a = None
        self._repoch_guard_trips = 0
        self._raw_align_mode = -1  # #343: provenance resets with the geom
        self._raw_plaus_prev_b = None   # #475 R2: gate state resets with geom
        self._raw_plaus_prev_t = None
        self._raw_plaus_consec = 0
        self._raw_plaus_held = 0
        self._r3_pair_rejects = 0       # #475 R3: acceptance state resets too
        self._r3_consec_rejects = 0
        self._r3_relocks = 0
        self._r4_synth_accepts = 0      # #475 R4: synthesis state resets too
        self._r4_synth_declines = 0
        self._r5_restamps = 0           # #475 R5
        self._sim_state_att_synth = False
        # T-240: re-assert the SIM_STATE rate-match each episode (idempotent;
        # the interval is per-boot link state and normally survives episode
        # resets — this guards mid-run autopilot reboots only). Witness
        # state (_rate_match_*) is per-BOOT, set at post-warmup init, and
        # deliberately NOT reset here.
        if (_FD_ENV_SIMSTATE_RATE_MATCH
                and getattr(self, "_mav", None) is not None):
            _send_sim_state_interval(self._mav)
        self._open_paired_csv()
        self._open_flight_state()
        self._prev_applied_residual = np.zeros(3, dtype=np.float32)
        self._prev_range_m = 0.0
        # v12 Commit 7: Learned-vs-baseline tracking
        self._m3b_residual_magnitudes: list = []
        self._m3b_prox_deltas: list = []
        self._m3a_fine_residual_magnitudes: list = []

        # RFL phase-gate hysteresis state
        self._rfl_active: bool = False
        self._rfl_activate_counter: int = 0
        self._rfl_deactivate_counter: int = 0
        if config.RESIDUAL_MODE == "reference" and self._ctrl_ref is not None:
            self._ctrl_ref.clear_position_offset()

        # Bias EMA state (episode-level, never reset mid-episode)
        self._bias_ema: float = 0.0
        self._voff_ema: float = 0.0  # VERTOFF-20260902: gated down-offset EMA, twin of _bias_ema
        # Extended logging accumulators
        self._c3o_policy_speed_delta_sum: float = 0.0
        self._c3o_policy_speed_delta_max: float = 0.0
        self._c3o_bias_speed_delta_sum: float = 0.0
        self._c3o_bias_active_ticks: int = 0
        self._c3o_bias_active_while_ahead_ticks: int = 0
        self._c3o_barrier_inside_ticks: int = 0
        self._c3o_barrier_outside_ticks: int = 0
        self._c3o_r_barrier_inside_sum: float = 0.0
        self._c3o_r_barrier_outside_sum: float = 0.0
        self._c3o_first_tier2_tick: int = -1
        self._c3o_first_bias_active_tick: int = -1
        self._c3o_ticks_to_90pct_bias: int = -1
        # Tier-4 hold-phase PSD instrumentation
        self._c3p_tier4_psd_values: list = []
        self._c3p_tier4_ticks: int = 0
        # Velocity modulation accumulator
        self._c3q_velocity_mod_values: list = []

        # v13 Commit 9: Per-episode reward accumulators (fix last-sub-tick overwrite)
        self._reward_accumulators = {
            'r_ref': 0.0, 'r_prox': 0.0, 'r_dwell': 0.0,
            'r_contact_dwell': 0.0,
            'r_safe': 0.0, 'r_smooth': 0.0, 'r_wake': 0.0,
            'r_m3a_coarse': 0.0, 'r_m3a_fine': 0.0,
            'r_range_penalty': 0.0, 'r_approach': 0.0,
            'r_action_rate': 0.0, 'r_exit': 0.0,
            'r_overspeed': 0.0, 'r_barrier': 0.0,
            'r_hold_fwd': 0.0,
            'r_ref_weighted': 0.0, 'r_prox_weighted': 0.0,
            'r_dwell_weighted': 0.0, 'r_safe_weighted': 0.0,
            'r_smooth_weighted': 0.0, 'r_wake_weighted': 0.0,
            'r_m3a_coarse_weighted': 0.0, 'r_m3a_fine_weighted': 0.0,
            'r_contact_dwell_weighted': 0.0,
            'r_action_rate_weighted': 0.0,
            'r_terminal_weighted': 0.0, 'r_approach_weighted': 0.0,
            'r_passthrough_weighted': 0.0,
            'r_barrier_weighted': 0.0, 'r_hold_fwd_weighted': 0.0,
            'r_range_penalty_weighted': 0.0, 'r_exit_weighted': 0.0,
            'r_overspeed_weighted': 0.0,
        }
        # OVSHAPE §5 item 2: per-episode aggregation for
        # the no-pass endpoint term. Key present ONLY when the flag is on so
        # flag-off keeps reward_components byte-identical (no stray
        # r_nopass_shape_mean). The endpoint value is accumulated ONCE per env
        # step at the seam (the sub-tick loop's .get() default contributes 0);
        # _mean therefore shares the sub-tick denominator with every other
        # accumulator. Max/min live in dedicated attrs, NOT in
        # _MRRL_RAW_COMPONENTS: that list is sampled per SUB-TICK with a 0.0
        # default, which would corrupt extrema for a once-per-step term.
        if self._nopass_shape_enable:
            self._reward_accumulators['r_nopass_shape'] = 0.0
        self._nopass_shape_max = float('-inf')
        self._nopass_shape_min = float('inf')
        # C006 Phase 7: Action residual accumulator for per-episode logging
        self._action_accum: list = []
        self._reward_subtick_count = 0

        # Phase E Batch 3a: per-episode min/max + explicit total accumulator
        # (operator M1: explicit total, not derived from weighted-component sum,
        # robust to future component additions).
        self._reward_max: Dict[str, float] = {
            k: float('-inf') for k in _MRRL_RAW_COMPONENTS
        }
        self._reward_min: Dict[str, float] = {
            k: float('inf') for k in _MRRL_RAW_COMPONENTS
        }
        self._reward_total_accumulator: float = 0.0
        # v13 Commit 13: Airspeed violation counter
        self._airspeed_violation_ticks = 0
        self._collision_ticks = 0
        self._nopass_ticks = 0  # NOPASSTERM-770
        # TERMDEMOTE-767: demoted-terminal events are PER-EPISODE, cleared
        # here beside the counters that feed them.
        self._demoted_events = {}
        # TELEMFIX-771: freshness state is PER-EPISODE. Carrying an age
        # across reset would let leg N+1 trust leg N's last message.
        self._telem_seen = {}
        self._telem_age_s = {}
        self._telem_abstains = {}
        # W-A: formation-lost latch re-arms fresh each episode
        self._formation_lost_armed = False

        # Build first observation
        obs = self._build_observation(dt_s=config.TICK_DT_S)
        # SHAPE-585: latch Phi(s0) from the SAME geometry the reset obs was
        # built from, so the live prev-chain mirrors the stored obs chain
        # exactly (buffer recompute uses obs[t] as prev for transition t).
        if self._ceiling_shape_enable:
            self._ceiling_prev_below_m = (
                self._last_geom.below_m if self._last_geom is not None else None)
        # OVSHAPE: latch Phi(s0) on the behind axis from the same geometry,
        # for the same reason (live prev-chain mirrors the stored obs chain).
        if self._nopass_shape_enable:
            self._nopass_prev_behind_m = (
                self._last_geom.behind_m if self._last_geom is not None else None)
        info: Dict[str, Any] = {"warmup_complete": True}

        # v18 Phase C: Mark warmup complete — step() assertion now safe to pass.
        self._warmup_complete = True

        # v7: Reset idle-drift counters and pause SITL until step() resumes it.
        # Between reset() return and the first step() call, SubprocVecEnv may
        # wait for other workers — SIGSTOP prevents drift during that gap.
        self._pause_count = 0
        self._unpause_count = 0
        self._max_pause_wall_s = 0.0
        self._pause_wall_start = None
        # v11: Reset sim-time correlation counters
        self._total_stopped_wall_s = 0.0
        self._last_sitl_sim_at_stop = 0.0
        self._last_sitl_sim_at_resume = 0.0
        self._pause_sitl()

        return obs, info

    @staticmethod
    def _parse_respid_bypass_crrl() -> bool:
        """RESPID §3 read site: unset ≡ "0"; only exactly "1" enables."""
        return os.environ.get("FD_RESPID_BYPASS_CRRL", "0").strip() == "1"

    def _compute_crrl_bounds(self, baseline) -> Optional[np.ndarray]:
        """P3-CRRL per-axis clip bounds; None disables the clip.

        None when CRRL is disabled OR the RESPID §3 probe bypass
        (FD_RESPID_BYPASS_CRRL=1) is active — the pre-existing
        crrl_bounds=None path in residual_action.process() skips ONLY the
        CRRL clip; every static clamp, EMA and slew stage remains.
        """
        if not config.CRRL_ENABLED or self._respid_bypass_crrl:
            return None
        speed_delta = baseline.speed_ms - self._follower_state.airspeed_ms
        alt_delta = baseline.altitude_m - self._follower_state.alt_msl_m
        if not config.CRRL_BPC_ENABLED:
            u_pid_delta = np.array([
                speed_delta,
                0.0,  # heading_delta — angular wraparound makes it ill-defined
                alt_delta,
            ], dtype=np.float64)
            return config.CRRL_BETA * np.abs(u_pid_delta) + config.CRRL_FLOOR
        # B'+C (bprime-c-v1; AD_ELECTION_INSTRUMENT.md §5 SIGNED Option A):
        # heading seam repaired — shortest-arc signed difference replaces the
        # hardcoded 0.0 (the T-63 authority-collapse seam). Per-sign bounds
        # [neg, pos]; elected terminal floors added only inside the RESPID-2
        # measurement window. Static clamps downstream still cap everything.
        heading_delta = self._wrap180(
            baseline.heading_deg - self._follower_state.heading_deg)
        u_abs = np.abs(np.array(
            [speed_delta, heading_delta, alt_delta], dtype=np.float64))
        beta_pos = np.asarray(config.CRRL_BPC_BETA, dtype=np.float64)
        # P-4 enabler: negative side may carry its own beta (defaults equal).
        beta_neg = np.asarray(config.CRRL_BPC_BETA_NEG, dtype=np.float64)
        state = getattr(self._ctrl_ref.state, "name", "") if self._ctrl_ref else ""
        in_win = 1.0 if state in config.CRRL_BPC_TERMINAL_STATES else 0.0
        pos = beta_pos * u_abs + in_win * np.asarray(
            config.CRRL_BPC_FLOOR_POS, dtype=np.float64)
        neg = beta_neg * u_abs + in_win * np.asarray(
            config.CRRL_BPC_FLOOR_NEG, dtype=np.float64)
        return np.stack([neg, pos])

    @staticmethod
    def _wrap180(angle_deg: float) -> float:
        """Signed shortest-arc angle in (-180, 180] (359->1 crossing = +2)."""
        return float(-((-angle_deg + 180.0) % 360.0 - 180.0))

    @staticmethod
    def _parse_respid2_inject() -> Optional[dict]:
        """RESPID-2 read site: FD_RESPID2_INJECT=
        '<axis>,<waveform>,<mag>[,<on_ticks>,<period_ticks>]'.

        axis in {speed|heading|alt|0|1|2}; waveform in {step|pulse}; mag clipped
        to [-1,1] (raw action units; raw==physical at action_scale=1.0). PULSE
        params are DECISION-TICKS (the driver converts from seconds via the
        fixed lockstep decision rate). Unset/empty => None => inert. FAILS LOUD
        on any malformed field (a silent typo would fly an inert probe and the
        adjudicator would falsely VOID). BANNED from campaign env lines.
        """
        raw = os.environ.get("FD_RESPID2_INJECT", "").strip()
        if not raw:
            return None
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 3:
            raise RuntimeError(
                "FD_RESPID2_INJECT needs axis,waveform,mag[,on_ticks,"
                f"period_ticks]: {raw!r}")
        axis_map = {"speed": 0, "heading": 1, "alt": 2, "0": 0, "1": 1, "2": 2}
        if parts[0].lower() not in axis_map:
            raise RuntimeError(
                f"FD_RESPID2_INJECT axis must be speed|heading|alt|0|1|2: {parts[0]!r}")
        if parts[1].lower() not in ("step", "pulse"):
            raise RuntimeError(
                f"FD_RESPID2_INJECT waveform must be step|pulse: {parts[1]!r}")
        try:
            mag = float(parts[2])
        except ValueError as e:
            raise RuntimeError(
                f"FD_RESPID2_INJECT magnitude unparseable: {parts[2]!r}") from e
        try:
            on_ticks = int(parts[3]) if len(parts) > 3 else 2
            period_ticks = int(parts[4]) if len(parts) > 4 else 4
        except ValueError as e:
            raise RuntimeError(
                f"FD_RESPID2_INJECT on_ticks/period_ticks must be ints: {raw!r}") from e
        if period_ticks < 1 or on_ticks < 0:
            raise RuntimeError(
                f"FD_RESPID2_INJECT needs period_ticks>=1, on_ticks>=0: {raw!r}")
        axis = axis_map[parts[0].lower()]
        if axis >= config.ACTION_DIM:
            raise RuntimeError(
                f"FD_RESPID2_INJECT axis {axis} >= ACTION_DIM {config.ACTION_DIM}")
        return {"axis": axis, "waveform": parts[1].lower(),
                "mag": float(np.clip(mag, -1.0, 1.0)),
                "on_ticks": on_ticks, "period_ticks": period_ticks}

    def _respid2_action(self) -> np.ndarray:
        """RESPID-2 phase-gated probe action (terminal window only).

        Returns a ZERO residual OUTSIDE {DOCKING_CREEP, HOLD_CONTACT}
        (baseline-only approach — the clean arm-Z trajectory) and the elected
        single-axis STEP/PULSE waveform INSIDE. Waveform time
        (``_respid2_win_tick``) resets on each window ENTRY, so a REJOIN
        excursion and re-entry restarts the probe clock.
        """
        spec = self._respid2_inject
        act = np.zeros(config.ACTION_DIM, dtype=np.float32)
        state = getattr(self._ctrl_ref.state, "name", "") if self._ctrl_ref else ""
        in_window = state in ("DOCKING_CREEP", "HOLD_CONTACT")
        if in_window:
            if not self._respid2_prev_in_win:
                self._respid2_win_tick = 0  # window-entry edge resets waveform time
            if spec["waveform"] == "pulse":
                phase = self._respid2_win_tick % spec["period_ticks"]
                val = spec["mag"] if phase < spec["on_ticks"] else 0.0
            else:  # step
                val = spec["mag"]
            act[spec["axis"]] = np.float32(np.clip(val, -1.0, 1.0))
            self._respid2_win_tick += 1
        self._respid2_prev_in_win = in_window
        return act

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute ACTION_REPEAT sub-ticks (4Hz DRL decision rate).

        Each call runs ``config.ACTION_REPEAT`` inner 20Hz ticks with the
        same action held constant, accumulating reward. Observation is
        built once at the end of the window (or on early termination).

        Returns (obs, reward, terminated, truncated, info).
        """
        # v18 Phase C: Prevent warmup transitions from entering replay buffer.
        assert self._warmup_complete, (
            f"step() called before warmup completed (env_id={self.env_id}). "
            f"This would contaminate the off-policy replay buffer with warmup data."
        )

        # === A2-r2 (T-281): deferred, NON-BLOCKING rate-match redundancy ===
        # r1 bought this send with time.sleep(0.5) at the handoff, which cost
        # 1.0 s of unobserved, uncommanded flight (G-1). Here it costs one
        # attribute check per step, cleared for the rest of the leg once fired.
        _due = getattr(self, "_rate_match_resend_due", None)
        if _due is not None and time.monotonic() >= _due:
            self._rate_match_resend_due = None
            if _send_sim_state_interval(self._mav):
                self._rate_match_sent += 1

        # === Phase 5c.A.6 ENTRYPOINT_DIAG (step_start) ===
        try:
            from . import entrypoint_diag as _ed
            if _ed.is_enabled():
                _ed.emit_event(
                    "step_start",
                    path="eval",
                    tick_source="step",
                    env_id=self.env_id,
                    step_idx=self._tick_count,
                    fsm_state=getattr(self._ctrl_ref.state, "name", "")
                        if self._ctrl_ref else "",
                )
        except Exception:
            pass

        # RESLOG-749: capture the policy's
        # action BEFORE any override, so the per-step residual log can
        # distinguish what the policy COMMANDED from what was APPLIED.
        _reslog_policy_action = np.asarray(action, dtype=np.float32).copy()

        # Inject-zero-action for normal-path identity proof.
        # P1 Item #7 (bundled hygiene fix): use config.ACTION_DIM rather
        # than the stale hardcoded `3` — pre-C3X remnant; ACTION_DIM = 2
        # under C3X Intervention 1 amputation.
        if config.INJECT_ZERO_ACTION:
            action = np.zeros(config.ACTION_DIM, dtype=np.float32)
        elif getattr(self, "_inject_const_action", None) is not None:
            # R1 rider: arm-R of the
            # residual-authority witness — a FIXED non-zero residual through
            # the IDENTICAL path the policy uses (this exact assignment site,
            # mirroring INJECT_ZERO_ACTION one line up). The witness is the
            # truth-side trajectory contrast between arm-Z and arm-R; a
            # readback is not a physical witness (#465 lesson). Values are
            # clipped to the action space at parse time; unset => byte-identical.
            action = self._inject_const_action.copy()
        elif getattr(self, "_respid2_inject", None) is not None:
            # RESPID-2: phase-gated bounded-terminal
            # probe — single-axis STEP/PULSE injected ONLY inside
            # {DOCKING_CREEP, HOLD_CONTACT}; action == 0 (baseline-only) outside
            # the window. Default OFF => byte-identical. BANNED from campaign
            # env lines. Spec: analysis/crrl_ad_20260724T234726Z/RESPID2_PREREG.md.
            action = self._respid2_action()

        # P1 Item #7 remediation: cache the effective raw action for
        # reward-computer passthrough-penalty logic. Captured AFTER the
        # INJECT_ZERO_ACTION override so identity-proof runs exercise the
        # penalty path as intended.
        self._last_raw_action = np.asarray(action, dtype=np.float32).copy()

        # C006 Phase 7: Accumulate raw action for per-episode residual logging
        if not hasattr(self, '_action_accum'):
            self._action_accum = []
        self._action_accum.append(action.copy())

        # RESLOG-749: per-step residual-action record. The source tag mirrors
        # the override-branch conditions above EXACTLY, without touching them.
        if config.INJECT_ZERO_ACTION:
            _reslog_source = "inject_zero"
        elif getattr(self, "_inject_const_action", None) is not None:
            _reslog_source = "inject_const"
        elif getattr(self, "_respid2_inject", None) is not None:
            _reslog_source = "respid2"
        else:
            _reslog_source = "policy"
        self._stash_residual_for_flight_state(_reslog_policy_action,
                                             _reslog_source)
        self._write_residual_action_row(
            _reslog_policy_action, action, _reslog_source)

        # v7: Resume SITL processes before processing this step.
        # _pump_mavlink() below naturally drains stale pre-freeze messages.
        self._unpause_sitl()

        # v12-fix: Reset stall detector wall-clock anchor after SIGCONT.
        # During on-policy training updates (PPO), the worker's time.monotonic()
        # advances (worker is NOT SIGSTOPped, only SITL is) while sim-time stays frozen.
        # Without this bump, the stall detector sees 200+s of wall-clock
        # with zero sim-time advance and fires a false STALL_SIMTIME.
        self._stall_detector.bump()

        # v13 Commit 12: Per-worker heartbeat (replaces global path).
        # Ensures watchdog sees per-env liveness even if main-process callback stalls.
        if self._tick_count % 100 == 0:
            self._heartbeat_path.touch(exist_ok=True)

        total_reward = 0.0
        terminated = False
        truncated = False
        terminal_type = ""
        reward_components: Dict[str, float] = {}
        dt_s = config.TICK_DT_S
        wake_intensity = 0.0
        in_m3b = False
        geom = self._last_geom  # fallback if no telemetry arrives
        sub_tick = 0
        # Must be set before compute() at line ~1006; computed at line ~819
        # on normal sub-ticks. This default covers SITL stall edge case (line ~631).
        _policy_delta = 0.0

        for sub_tick in range(config.ACTION_REPEAT):
            self._tick_count += 1

            # --- Drain MAVLink and update telemetry ---
            self._pump_mavlink()

            # --- D2 (FD_CURRICULUM_V2): GPS degradation ladder (env-side) ---
            # Slow GLTCH walk at ~the GPS rate + dropout toggling. Inert when the
            # flag is off (walk is None) or unset (partial-mock envs). The fuser
            # is never touched. getattr keeps the legacy/mock step path identical.
            if getattr(self, "_gps_walk", None) is not None:
                self._apply_gps_ladder()

            # --- v12: SITL liveness + stall detection ---
            if self._sitl_proc is not None and self._sitl_proc.poll() is not None:
                logger.warning(
                    f"[DRL env_id={self.env_id}] SITL process exited "
                    f"(returncode={self._sitl_proc.returncode}) — STALL_SIMTIME"
                )
                terminal_type = "STALL_SIMTIME"
                terminated = True
                self._stall_count += 1
                break

            if self._stall_detector.check(self._time_auth.now_s()):
                logger.warning(
                    f"[DRL env_id={self.env_id}] Wall-clock stall detected "
                    f"(sim_time frozen for 30s) — STALL_SIMTIME"
                )
                terminal_type = "STALL_SIMTIME"
                terminated = True
                self._stall_count += 1
                break

            # --- Compute geometry ---
            geom = self._compute_geometry()
            # item 9: first-boot only; getattr-safe because mocked tests bypass
            # __init__ (same pattern as _belly_prev_alt_cmd, env:782).
            if getattr(self, "_align_census_state", None) is not None:
                self._align_census_tick()
            if _FD_T0SL_PROBE:  # module-global gate (robust to partial __init__ in tests)
                self._t0sl_poison_truth()  # canary-poison truth (probe mode only)
            if geom is None:
                if sub_tick == 0:
                    # No telemetry on first sub-tick — bail immediately
                    obs = np.zeros(config.OBS_DIM, dtype=np.float32)
                    return obs, 0.0, False, False, {"error": "no_telemetry"}
                # Mid-window telemetry gap — keep going with last known geom
                geom = self._last_geom
                if geom is None:
                    break

            # --- Wake turbulence check ---
            wake_intensity = 0.0
            if self._wake_model is not None:
                leader_gs = math.sqrt(self._leader_vn ** 2 + self._leader_ve ** 2)
                time_since = (
                    geom.behind_m / max(leader_gs, 1e-3) if geom.behind_m > 0 else 0.0
                )
                _, wake_details = self._wake_model.is_in_avoid_zone(
                    geom, time_since, (0.0, 0.0, 0.0),
                )
                wake_intensity = wake_details["hazard_intensity"]

            # --- Phase and targets ---
            target_behind, target_below = self._get_phase_targets(geom)

            # T-438: snapshot the breakaway-guard counters BEFORE this
            # sub-tick's tick_once(); differenced after the modal block.
            # getattr: partial-mock envs (__new__, no __init__) default OFF.
            _rsafe_on = getattr(self, "_rsafe_wire_enable", False)
            _rsafe_g0 = self._rsafe_guard_fires_now() if _rsafe_on else 0

            # --- Command pipeline (modal: reference vs actuator) ---
            if config.RESIDUAL_MODE == "reference":
                # === REFERENCE MODE — phase-gated geometry offsets ===

                # Phase-gate with hysteresis: activate after N ticks in M3a_fine,
                # deactivate after M ticks below M3a_coarse.
                if config.FORCE_ZERO_RESIDUAL:
                    # Hard-zero override: bypass model entirely (Phase 0 re-anchor)
                    self._ctrl_ref.clear_position_offset()
                    applied_residual = np.zeros(3, dtype=np.float32)
                elif self._rfl_active:
                    # RFL is active — check deactivation condition
                    if self._milestone_tier < 2:  # Below M3a_coarse
                        self._rfl_deactivate_counter += 1
                        self._rfl_activate_counter = 0
                        if self._rfl_deactivate_counter >= 20:  # 1.0s at 20Hz
                            self._rfl_active = False
                            self._rfl_deactivate_counter = 0
                    else:
                        self._rfl_deactivate_counter = 0
                    if self._rfl_active:
                        offset_result = self._action_proc.process_reference(
                            action, action_scale=self._action_scale,
                            tick_count=self._tick_count,
                        )
                        self._ctrl_ref.set_position_offset(
                            offset_result.applied_residual[0],
                            offset_result.applied_residual[1],
                            offset_result.applied_residual[2],
                        )
                        applied_residual = offset_result.applied_residual.copy()
                    else:
                        self._ctrl_ref.clear_position_offset()
                        applied_residual = np.zeros(3, dtype=np.float32)
                else:
                    # RFL is inactive — check activation condition
                    if self._milestone_tier >= config.RFL_ACTIVATION_TIER:
                        self._rfl_activate_counter += 1
                        self._rfl_deactivate_counter = 0
                        if self._rfl_activate_counter >= 10:  # 0.5s at 20Hz
                            self._rfl_active = True
                            self._rfl_activate_counter = 0
                    else:
                        self._rfl_activate_counter = 0
                    self._ctrl_ref.clear_position_offset()
                    applied_residual = np.zeros(3, dtype=np.float32)

                # Controller ticks with true OR modified geometry
                baseline_cmd = self._ctrl_ref.tick_once()
                if baseline_cmd is not None:
                    self._last_baseline_cmd = baseline_cmd
                else:
                    baseline_cmd = self._last_baseline_cmd

                # Send baseline commands DIRECTLY (no actuator-level addition).
                # #452: belly clamp wraps the realized altitude at the env
                # chokepoint (covers reference mode, which residual_action cannot).
                self._send_commands(
                    baseline_cmd.speed_ms, baseline_cmd.heading_deg,
                    self._apply_belly_clamp(
                        self._apply_faultinj_vz(baseline_cmd.altitude_m, dt_s), dt_s),
                )

                # For reward/obs compatibility: use ProcessedAction-like applied_residual
                class _RefResult:
                    pass
                result = _RefResult()
                result.applied_residual = applied_residual

                # policy_speed_delta accumulator (reference-mode)
                _policy_delta = applied_residual[0]
                self._c3o_policy_speed_delta_sum += _policy_delta
                self._c3o_policy_speed_delta_max = max(
                    self._c3o_policy_speed_delta_max, abs(_policy_delta))
                if self._milestone_tier >= 4:
                    self._c3p_tier4_psd_values.append(_policy_delta)
                    self._c3p_tier4_ticks += 1

            else:
                # === ACTUATOR MODE ===
                baseline_cmd = self._ctrl_ref.tick_once()
                if baseline_cmd is not None:
                    self._last_baseline_cmd = baseline_cmd
                else:
                    baseline_cmd = self._last_baseline_cmd

                baseline = baseline_cmd

                # === Phase 5c.A.6 ENTRYPOINT_DIAG (step_inner_tick) ===
                try:
                    from . import entrypoint_diag as _ed
                    if _ed.is_enabled():
                        _ed.emit_event(
                            "step_inner_tick",
                            path="eval",
                            tick_source="step",
                            env_id=self.env_id,
                            step_idx=self._tick_count,
                            fsm_state=getattr(self._ctrl_ref.state, "name", ""),
                            speed_cmd=getattr(baseline, "speed_ms", "") if baseline else "",
                            alt_cmd=getattr(baseline, "altitude_m", "") if baseline else "",
                            command_type="captured" if baseline_cmd is not None else "stale",
                        )
                except Exception:
                    pass

                # Bumpless transfer verification (first DRL tick only)
                if self._tick_count == 1:
                    speed_delta = abs(baseline.speed_ms - getattr(self,
                                      '_warmup_last_speed', baseline.speed_ms))
                    hdg_ref = getattr(self, '_warmup_last_heading', None)
                    if hdg_ref is not None:
                        heading_delta = baseline.heading_deg - hdg_ref
                        while heading_delta > 180:
                            heading_delta -= 360
                        while heading_delta < -180:
                            heading_delta += 360
                        heading_delta = abs(heading_delta)
                    else:
                        heading_delta = 0.0
                    if speed_delta > 0.5 or heading_delta > 2.0:
                        logger.warning(
                            f"[DRL env_id={self.env_id}] BUMPLESS_TRANSFER_DISCONTINUITY: "
                            f"speed_delta={speed_delta:.2f} m/s, heading_delta={heading_delta:.2f} deg"
                        )

                # P3-CRRL: PID-delta proportional residual clipping
                # ACTION_DIM=3 (cross-axis restored 2026-05-06):
                # 3-element [speed_delta, heading_delta, altitude_delta]. Heading
                # PID-delta uses 0.0 (angular wraparound makes a meaningful
                # delta ill-defined here); CRRL_FLOOR provides the minimum
                # heading clamp bound, and CLAMP_HEADING_*_DEG inside the
                # residual processor caps the per-tick magnitude.
                crrl_bounds = self._compute_crrl_bounds(baseline)

                # Force-zero in actuator mode: bypass model output
                if config.FORCE_ZERO_RESIDUAL:
                    # Pin all filter/bias state to exact zero (not converging)
                    self._bias_ema = 0.0
                    self._voff_ema = 0.0
                    self._action_proc.pin_zero()
                    from .residual_action import ProcessedAction
                    result = ProcessedAction(
                        speed_ms=baseline.speed_ms,
                        heading_deg=baseline.heading_deg,
                        altitude_m=baseline.altitude_m,
                        applied_residual=np.zeros(3, dtype=np.float32),
                    )
                else:
                    # Compute phase-gated forward bias
                    if config.FORWARD_BIAS_MS <= 0.0:
                        _gated_bias = 0.0
                    elif self._milestone_tier >= 2:
                        # Bias active at tier >= 2, persists through terminal
                        _target_bias = config.FORWARD_BIAS_MS
                        self._bias_ema = (config.BIAS_SMOOTH_ALPHA * _target_bias
                                          + (1.0 - config.BIAS_SMOOTH_ALPHA) * self._bias_ema)
                        _gated_bias = self._bias_ema
                    else:
                        # Before M2: decay bias EMA toward zero
                        self._bias_ema = ((1.0 - config.BIAS_SMOOTH_ALPHA) * self._bias_ema)
                        _gated_bias = self._bias_ema

                    # VERTOFF-20260902: gated down-offset, exact twin of the bias
                    # gate above (tier >= 2, persists through terminal, same EMA).
                    if config.VERT_OFFSET_DOWN_M <= 0.0:
                        _gated_voff = 0.0
                    elif self._milestone_tier >= 2:
                        self._voff_ema = (config.BIAS_SMOOTH_ALPHA * config.VERT_OFFSET_DOWN_M
                                          + (1.0 - config.BIAS_SMOOTH_ALPHA) * self._voff_ema)
                        _gated_voff = self._voff_ema
                    else:
                        self._voff_ema = ((1.0 - config.BIAS_SMOOTH_ALPHA) * self._voff_ema)
                        _gated_voff = self._voff_ema

                    result = self._action_proc.process(
                        baseline, action, geom.range_m,
                        self._leader_state.alt_msl_m, dt_s,
                        action_scale=self._action_scale,
                        tick_count=self._tick_count,
                        crrl_bounds=crrl_bounds,
                        forward_bias=_gated_bias,
                        vert_offset_down_m=_gated_voff,
                    )

                    # Extended logging accumulators
                    _policy_delta = result.applied_residual[0] - _gated_bias
                    self._c3o_policy_speed_delta_sum += _policy_delta
                    self._c3o_policy_speed_delta_max = max(
                        self._c3o_policy_speed_delta_max, abs(_policy_delta))
                    # Tier-4 hold-phase PSD accumulator
                    if self._milestone_tier >= 4:
                        self._c3p_tier4_psd_values.append(_policy_delta)
                        self._c3p_tier4_ticks += 1
                    self._c3o_bias_speed_delta_sum += _gated_bias
                    if _gated_bias > 0.01:
                        self._c3o_bias_active_ticks += 1
                        if self._c3o_first_bias_active_tick < 0:
                            self._c3o_first_bias_active_tick = self._tick_count
                    if self._milestone_tier >= 2 and self._c3o_first_tier2_tick < 0:
                        self._c3o_first_tier2_tick = self._tick_count
                    if (self._c3o_ticks_to_90pct_bias < 0
                            and self._bias_ema >= 0.9 * config.FORWARD_BIAS_MS
                            and config.FORWARD_BIAS_MS > 0
                            and self._c3o_first_tier2_tick >= 0):
                        self._c3o_ticks_to_90pct_bias = (
                            self._tick_count - self._c3o_first_tier2_tick)

                self._send_commands(result.speed_ms, result.heading_deg,
                                    self._apply_belly_clamp(
                                        self._apply_faultinj_vz(result.altitude_m, dt_s), dt_s))

            # --- Update dwell tracking ---
            behind_err, cross_err, below_err = EnvelopeChecker.get_envelope_errors(
                geom, self._leader_state.alt_msl_m, self._follower_state.alt_msl_m,
                target_behind, 0.0, target_below,
            )

            in_m3b = self._check_m3b(behind_err, cross_err, below_err)
            if in_m3b:
                self._continuous_m3b_dwell_s += dt_s
                self._accumulated_m3b_dwell_s += dt_s
            else:
                self._continuous_m3b_dwell_s = 0.0

            # --- Milestone tier tracking ---
            self._milestone_tier = self._compute_milestone_tier(
                behind_err, cross_err, below_err,
            )
            # v12: Track M3a band membership for graduated rewards
            in_m3a_coarse = self._milestone_tier >= 2
            in_m3a_fine = self._milestone_tier >= 3
            if in_m3a_fine:
                self._continuous_m3a_fine_dwell_s += dt_s
            else:
                self._continuous_m3a_fine_dwell_s = 0.0

            episode_sim_s = self._time_auth.now_s() - self._episode_start_sim_s
            for t in range(1, self._milestone_tier + 1):
                if t not in self._tier_first_achieved_s:
                    self._tier_first_achieved_s[t] = episode_sim_s
                self._tier_dwell_s[t] = self._tier_dwell_s.get(t, 0.0) + dt_s
            if geom is not None:
                self._min_range_m = min(self._min_range_m, geom.range_m)

            # --- v19: True FlightDock contact milestones (boom-tip reference) ---
            # Uses geom.behind_m/below_m/cross_m directly (already boom-tip-
            # referenced per geometry contract).
            _prev_contact_m3b_dwell = self._continuous_contact_m3b_dwell_s
            in_contact_m3b = False  # Default for geom=None path
            if geom is not None:
                in_contact_m3a = self._check_contact_m3a(geom)
                in_contact_m3b = self._check_contact_m3b(geom)
                if in_contact_m3b:
                    self._continuous_contact_m3b_dwell_s += dt_s
                    self._accumulated_contact_m3b_dwell_s += dt_s
                else:
                    self._best_contact_m3b_dwell_s = max(
                        self._best_contact_m3b_dwell_s,
                        self._continuous_contact_m3b_dwell_s,
                    )
                    self._continuous_contact_m3b_dwell_s = 0.0
                if in_contact_m3a:
                    self._continuous_contact_m3a_dwell_s += dt_s
                    if self._continuous_contact_m3a_dwell_s >= config.CONTACT_M3A_DWELL_S:
                        self._contact_m3a_achieved = True
                else:
                    self._continuous_contact_m3a_dwell_s = 0.0
                if geom.range_m == geom.range_m:  # NaN guard
                    self._boom_tip_min_range_m = min(
                        self._boom_tip_min_range_m, geom.range_m,
                    )

                # C006 P4: Actual M2 (follower at 10m ±2m behind/below boom tip)
                in_actual_m2 = (
                    abs(geom.behind_m - config.ACTUAL_M2_BEHIND_TARGET)
                    <= config.ACTUAL_M2_BEHIND_TOL
                    and abs(geom.cross_m) < config.ACTUAL_M2_CROSS_MAX
                    and abs(geom.below_m - config.ACTUAL_M2_BELOW_TARGET)
                    <= config.ACTUAL_M2_BELOW_TOL
                )
                if in_actual_m2:
                    self._continuous_actual_m2_dwell_s += dt_s
                else:
                    self._best_actual_m2_dwell_s = max(
                        self._best_actual_m2_dwell_s,
                        self._continuous_actual_m2_dwell_s,
                    )
                    self._continuous_actual_m2_dwell_s = 0.0

                # C006 P4: M3a Coarse (±2.0m behind/below, ±1.5m cross from boom tip)
                in_contact_m3a_coarse = (
                    abs(geom.behind_m) <= config.CONTACT_M3A_COARSE_BEHIND_TOL
                    and abs(geom.below_m) <= config.CONTACT_M3A_COARSE_BELOW_TOL
                    and abs(geom.cross_m) <= config.CONTACT_M3A_COARSE_CROSS_TOL
                )
                if in_contact_m3a_coarse:
                    self._continuous_contact_m3a_coarse_dwell_s += dt_s
                    if (self._continuous_contact_m3a_coarse_dwell_s
                            >= config.CONTACT_M3A_COARSE_DWELL_S):
                        self._contact_m3a_coarse_achieved = True
                else:
                    self._continuous_contact_m3a_coarse_dwell_s = 0.0

            # --- Dwell-failure decomposition — dominant exit axis at break sample ---
            _r_exit_this_subtick = 0.0
            if in_contact_m3b and not self._m3b_was_in_contact:
                self._m3b_entry_count += 1
                # Record TTC at M3b entry for diagnostics
                if geom is not None and geom.closure_rate_ms > 0.01:
                    _ttc_entry = (geom.behind_m + config.CONTACT_M3B_BEHIND_TOL) / geom.closure_rate_ms
                    self._ttc_at_m3b_entries.append(float(np.clip(_ttc_entry, -5.0, 5.0)))
            elif not in_contact_m3b and self._m3b_was_in_contact:
                self._m3b_break_count += 1
                self._m3b_total_dwell_before_break_s += _prev_contact_m3b_dwell
                if geom is not None:
                    # geom reflects the current sub-tick state after MAVLink update.
                    # On the exit sub-tick, this is the first out-of-contact geometry,
                    # so margins may already be negative. This is the intended behavior:
                    # we measure how far past the boundary the agent traveled.
                    behind_margin = config.CONTACT_M3B_BEHIND_TOL - abs(geom.behind_m)
                    cross_margin = config.CONTACT_M3B_CROSS_TOL - abs(geom.cross_m)
                    below_margin = config.CONTACT_M3B_BELOW_TOL - abs(geom.below_m)
                    margins = {"behind": behind_margin, "cross": cross_margin,
                               "below": below_margin}
                    # Most-violated axis at the exit sample, not necessarily the first
                    # axis to cross the boundary during the exit transition. For true
                    # first-break detection, per-subtick boundary latching would be
                    # required. This approximation is sufficient for diagnostics.
                    break_axis = min(margins, key=margins.get)
                    self._m3b_last_break_axis = break_axis
                    self._m3b_last_break_margin_m = margins[break_axis]
                    self._m3b_break_axis_counts[break_axis] += 1
                    if break_axis == "behind":
                        self._m3b_last_break_velocity_mps = abs(geom.closure_rate_ms)
                    elif break_axis == "cross":
                        self._m3b_last_break_velocity_mps = abs(geom.vel_cross_m_s)
                    elif break_axis == "below":
                        self._m3b_last_break_velocity_mps = abs(geom.vel_below_m_s)
                else:
                    # Geometry unavailable at break — record but cannot attribute axis
                    self._m3b_last_break_axis = "unknown"
                    self._m3b_last_break_margin_m = -999.0
                    self._m3b_last_break_velocity_mps = -1.0
                # Campaign 3h: exit penalty for premature M3b departure
                # Lower gate filters noise/boundary crossings (<0.5s).
                # Upper threshold defines "premature" (<1.5s).
                if (config.EXIT_DWELL_LOWER_GATE_S
                        <= _prev_contact_m3b_dwell
                        < config.EXIT_DWELL_THRESHOLD_S):
                    _r_exit_this_subtick = config.R_EXIT_PENALTY
                    # Behind-axis exits get amplified penalty
                    if break_axis == "behind":
                        _r_exit_this_subtick *= config.EXIT_BEHIND_MULTIPLIER
                        self._behind_axis_exit_count += 1
                        self._r_exit_behind_total += _r_exit_this_subtick
                    else:
                        self._non_behind_exit_count += 1
                    self._penalized_exit_count += 1
                self._pre_exit_dwell_accum += _prev_contact_m3b_dwell
                self._pre_exit_dwell_count += 1
            self._m3b_was_in_contact = in_contact_m3b

            if behind_err < 0:
                self._max_behind_overshoot_m = max(
                    self._max_behind_overshoot_m, abs(behind_err),
                )
            self._max_cross_excursion_m = max(
                self._max_cross_excursion_m, abs(cross_err),
            )

            # T-455: per-episode vertical extrema on BOTH channels + the
            # guard-fire delta. TELEMETRY ONLY -- see the declaration block at
            # __init__ for the G4 containment contract and the 10 Hz-hold
            # warning. Read by the info dict and by nothing else.
            if geom is not None:
                self._max_above_boom_fused_m = max(
                    self._max_above_boom_fused_m, max(0.0, -geom.below_m),
                )
            # Truth updates ONLY on a live sample: a None tick must leave the
            # extremum untouched, never reset it to a default (FM-6). Same
            # freshness contract as the collision judge, which reads this
            # attribute a few lines later in _check_termination.
            # getattr, not attribute access: "no truth sample" and "attribute
            # absent" are the same condition for this instrument, and both must
            # leave the extremum untouched.
            _tg = getattr(self, "_raw_truth_geom", None)
            if _tg is not None:
                self._max_above_boom_truth_m = max(
                    self._max_above_boom_truth_m, max(0.0, -_tg.below_m),
                )
                if _tg.range_m == _tg.range_m:      # NaN guard
                    self._min_boom_tip_range_truth_m = min(
                        self._min_boom_tip_range_truth_m, _tg.range_m,
                    )
            # DELTA, never the controller's cumulative count: the baseline is
            # latched on the first tick of the episode that can see the
            # controller. Perception-side counters (G4-clean) and independent
            # of FD_RSAFE_WIRE_ENABLE, so a guard-fire rate is recorded whether
            # or not r_safe is pricing it.
            _gnow = self._rsafe_guard_fires_now()
            if self._guard_fires_base is None:
                self._guard_fires_base = _gnow
            self._guard_fires_episode = max(0, _gnow - self._guard_fires_base)

            # --- Termination check ---
            terminated, truncated, terminal_type = self._check_termination()

            # T-438: per-sub-tick breakaway-guard fire DELTA (clamped >= 0);
            # 0 with the flag off (default) — the historical literal.
            _rsafe_fires = (max(0, self._rsafe_guard_fires_now() - _rsafe_g0)
                            if _rsafe_on else 0)

            # --- Reward (accumulated across sub-ticks) ---
            sub_reward, reward_components = self._reward_comp.compute(
                behind_error_m=behind_err,
                cross_error_m=cross_err,
                below_error_m=below_err,
                applied_residual=result.applied_residual,
                prev_applied_residual=self._prev_applied_residual,
                in_m3b=in_m3b,
                continuous_dwell_s=self._continuous_m3b_dwell_s,
                guard_fires=_rsafe_fires,      # T-438: 0 unless flag ON
                safety_violations=0,           # elected 2026-08-11: stays 0
                is_terminal=terminated or truncated,
                terminal_type=terminal_type,
                wake_hazard_intensity=wake_intensity,
                in_m3a_coarse=in_m3a_coarse,
                in_m3a_fine=in_m3a_fine,
                continuous_m3a_fine_dwell_s=self._continuous_m3a_fine_dwell_s,
                boom_tip_range_m=geom.range_m,
                curriculum_stage=self._curriculum_stage,
                in_contact_m3b=in_contact_m3b,
                continuous_contact_m3b_dwell_s=self._continuous_contact_m3b_dwell_s,
                dt_s=dt_s,
                milestone_tier=self._milestone_tier,
                closure_rate_ms=geom.closure_rate_ms if geom is not None else 0.0,
                policy_speed_delta=_policy_delta,
                raw_action=self._last_raw_action,
            )
            try:                       # R0: mocked/partial-init paths
                self._mrrl_keys_seen.update(reward_components)
            except AttributeError:     # never block a step on telemetry
                self._mrrl_keys_seen = set(reward_components)

            # Barrier inside/outside logging
            _r_barrier_this = reward_components.get("r_barrier", 0.0)
            if _r_barrier_this != 0.0:
                if in_m3b:
                    self._c3o_barrier_inside_ticks += 1
                    self._c3o_r_barrier_inside_sum += _r_barrier_this
                else:
                    self._c3o_barrier_outside_ticks += 1
                    self._c3o_r_barrier_outside_sum += _r_barrier_this
            # Accumulate velocity modulation when inside precontact hold zone
            _velocity_mod = reward_components.get("velocity_mod", 1.0)
            if in_m3b:
                self._c3q_velocity_mod_values.append(_velocity_mod)
            # Bias-while-ahead diagnostic
            if hasattr(self, '_c3o_bias_speed_delta_sum'):
                _current_bias = getattr(self, '_bias_ema', 0.0)
                if _current_bias > 0.01 and behind_err < 0:
                    self._c3o_bias_active_while_ahead_ticks += 1

            range_penalty = self._compute_range_penalty()
            sub_reward += range_penalty
            reward_components["r_range_penalty"] = range_penalty
            reward_components["r_range_penalty_weighted"] = range_penalty
            # Campaign 3h: exit penalty (weighted)
            weighted_exit = _r_exit_this_subtick * self._reward_comp.w_exit_penalty
            sub_reward += weighted_exit
            reward_components["r_exit"] = _r_exit_this_subtick
            reward_components["r_exit_weighted"] = weighted_exit
            # Dynamic overspeed penalty in terminal corridor
            _r_overspeed_this_subtick = 0.0
            if (geom is not None
                    and config.OVERSPEED_CORRIDOR_INNER < geom.behind_m <= config.OVERSPEED_CORRIDOR_OUTER
                    and geom.closure_rate_ms > 0):
                _corridor_range = config.OVERSPEED_CORRIDOR_OUTER - config.OVERSPEED_CORRIDOR_INNER
                _fraction = (geom.behind_m - config.OVERSPEED_CORRIDOR_INNER) / _corridor_range
                _v_limit = config.OVERSPEED_V_MIN + _fraction * (config.OVERSPEED_V_MAX - config.OVERSPEED_V_MIN)
                _v_excess = max(0.0, geom.closure_rate_ms - _v_limit)
                if _v_excess > 0:
                    _r_overspeed_this_subtick = -1.0 * _v_excess
                    self._overspeed_tick_count += 1
                    self._max_v_excess = max(self._max_v_excess, _v_excess)
            # Gate off overspeed penalty inside precontact hold zone
            if in_m3b:
                _r_overspeed_this_subtick = 0.0
            weighted_overspeed = _r_overspeed_this_subtick * self._reward_comp.w_overspeed
            sub_reward += weighted_overspeed
            reward_components["r_overspeed"] = _r_overspeed_this_subtick
            reward_components["r_overspeed_weighted"] = weighted_overspeed
            self._r_overspeed_total += _r_overspeed_this_subtick
            total_reward += sub_reward

            # Track approach suppression and closure_rate_delta in M3b
            if in_contact_m3b and reward_components.get("r_approach", 0.0) == 0.0:
                self._r_approach_suppressed_ticks += 1
                self._approach_suppression_active = True
            if in_contact_m3b and geom is not None:
                # Record closure_rate_delta while in M3b for diagnostics
                obs_builder_crd = getattr(self._obs_builder, '_prev_closure_rate_ms', 0.0)
                if obs_builder_crd != 0.0:
                    crd = (geom.closure_rate_ms - obs_builder_crd) / max(dt_s, 0.001)
                    self._closure_rate_delta_in_m3b.append(crd)

            # v13 Commit 9: Accumulate all 9 components every sub-tick
            for _k in self._reward_accumulators:
                self._reward_accumulators[_k] += reward_components.get(_k, 0.0)
            self._reward_subtick_count += 1

            # Phase E Batch 3a: per-episode min/max + explicit total accumulation.
            # Telemetry-only; no reward-path side effects.
            # NOTE: r_terminal / r_passthrough excluded from raw aggregator per Batch 3a-bis
            # null-finding (Item 1); their weighted forms (r_terminal_weighted,
            # r_passthrough_weighted) are surfaced via reward_components dict instead.
            for _tk in _MRRL_RAW_COMPONENTS:
                _tv = reward_components.get(_tk, 0.0)
                if _tv > self._reward_max[_tk]:
                    self._reward_max[_tk] = _tv
                if _tv < self._reward_min[_tk]:
                    self._reward_min[_tk] = _tv
            self._reward_total_accumulator += sub_reward

            # --- v12 Commit 7: Learned-vs-baseline residual tracking ---
            raw_residual_mag = float(np.mean(np.abs(action)))
            if self._milestone_tier >= 4:  # M3b
                self._m3b_residual_magnitudes.append(raw_residual_mag)
                self._m3b_prox_deltas.append(reward_components.get("r_prox", 0.0))
            if self._milestone_tier >= 3:  # M3a fine
                self._m3a_fine_residual_magnitudes.append(raw_residual_mag)

            # --- Update state for next sub-tick ---
            self._prev_applied_residual = result.applied_residual.copy()
            self._obs_builder.set_prev_applied_residual(result.applied_residual)
            if geom is not None:
                self._prev_range_m = geom.range_m

            # Patch C2 2026-05-06: paired-channel CSV row (RAW / ENV / CTRL).
            self._write_paired_row(env_geom=geom, in_m3b_env=in_m3b)
            # TELEM-766-A: full flight-state record. A SEPARATE call, not
            # folded into _write_paired_row -- that returns early when its
            # own file is absent, and a logging failure must not cascade.
            self._write_flight_state_row(env_geom=geom,
                                         terminal_reason=terminal_type)

            # --- Tick pacing ---
            self._tick_sleep(dt_s)

            if terminated or truncated:
                break

        # v19: Flush contact M3b best-dwell at episode end (in case the episode
        # terminates while still inside the M3b envelope — the else-branch that
        # normally flushes only fires on envelope exit).
        if terminated or truncated:
            self._best_contact_m3b_dwell_s = max(
                self._best_contact_m3b_dwell_s,
                self._continuous_contact_m3b_dwell_s,
            )

        # --- Build observation at END of action-repeat window ---
        obs = self._build_observation(dt_s, wake_hazard_intensity=wake_intensity)

        # SHAPE-585: endpoint ceiling shaping, ONCE per env step on the same
        # END-of-window perception geometry the obs was just built from
        # (obs[2] = below_m/50) — this is what makes the term bit-exactly
        # recomputable from stored replay-buffer endpoints. Default-OFF:
        # flag-off leaves this step byte-identical. G4: perception geometry
        # only (self._last_geom), truth never enters.
        # getattr defaults (not bare attrs): the unit-test fixtures build this
        # env via __new__ without __init__ — the T-438 call site (rsafe, :2472)
        # set the precedent; default False == flag-off == byte-identical.
        if getattr(self, "_ceiling_shape_enable", False) and self._last_geom is not None:
            _curr_below = self._last_geom.below_m
            _prev_below = getattr(self, "_ceiling_prev_below_m", None)
            if _prev_below is None:
                _prev_below = _curr_below
            _r_ceiling = ceiling_shaping_reward(_prev_below, _curr_below)
            total_reward += _r_ceiling
            reward_components["r_ceiling_shape"] = _r_ceiling
            self._ceiling_prev_below_m = _curr_below

        # OVSHAPE-20260816: endpoint no-pass shaping, same seam and contract
        # as the SHAPE-585 block above, behind axis (obs[0] = behind_m/50).
        # Extracted to a method so the flag-off byte-identity is provable by a
        # RUNTIME differential, not source inspection alone. Default-OFF.
        total_reward = self._apply_nopass_shape(total_reward, reward_components)

        # Record actual accumulated reward (all algorithms see total_reward, not last sub-tick)
        reward_components["total_step"] = total_reward

        # v13 Commit 9: Per-episode means for CSV logging (replaces last-sub-tick snapshot)
        n = max(self._reward_subtick_count, 1)
        for _k, _v in self._reward_accumulators.items():
            reward_components[f"{_k}_mean"] = _v / n

        # Phase E Batch 3a: per-episode max / min / total_mean / pathology ratios.
        for _tk in _MRRL_RAW_COMPONENTS:
            _tmax = self._reward_max[_tk]
            _tmin = self._reward_min[_tk]
            # Guard against episodes with zero sub-ticks (min/max sentinels still at inf).
            reward_components[f"{_tk}_max"] = _tmax if _tmax != float('-inf') else 0.0
            reward_components[f"{_tk}_min"] = _tmin if _tmin != float('inf') else 0.0
        # OVSHAPE §5 item 2: extrema for the endpoint term from its dedicated
        # attrs (kept out of _MRRL_RAW_COMPONENTS — see reset comment).
        # Emitted only when the flag is on, so flag-off stays byte-identical.
        if getattr(self, "_nopass_shape_enable", False):
            _nmax = getattr(self, "_nopass_shape_max", float('-inf'))
            _nmin = getattr(self, "_nopass_shape_min", float('inf'))
            reward_components["r_nopass_shape_max"] = (
                _nmax if _nmax != float('-inf') else 0.0)
            reward_components["r_nopass_shape_min"] = (
                _nmin if _nmin != float('inf') else 0.0)
        total_mean = self._reward_total_accumulator / n
        reward_components["total_mean"] = total_mean
        # Pathology ratios (passive diagnostics):
        #   milestone/reference: dense-reward-dominance signal
        #   |smoothness|/|total|: jitter / braking-exploitation proxy
        _eps = 1e-9
        r_ref_m = reward_components.get("r_ref_mean", 0.0)
        r_m3a_c = reward_components.get("r_m3a_coarse_mean", 0.0)
        r_m3a_f = reward_components.get("r_m3a_fine_mean", 0.0)
        r_smooth_m = reward_components.get("r_smooth_mean", 0.0)
        reward_components["ratio_milestone_to_ref_mean"] = (
            (r_m3a_c + r_m3a_f) / (abs(r_ref_m) + _eps)
        )
        reward_components["ratio_smoothness_abs_to_total_mean"] = (
            abs(r_smooth_m) / (abs(total_mean) + _eps)
        )

        info: Dict[str, Any] = {
            "reward_components": reward_components,
            "leader_profile": self._current_leader_profile,
            "wind_speed_ms": self._current_wind_speed_ms,
            "wind_dir_deg": self._current_wind_dir_deg,
            # T-393: wind_turb is the LEGACY knob and is INERT on the json
            # backend -- kept for continuity, never reported alone. The flown
            # turbulence is wind_sigma_ms (FDM Dryden triple).
            "wind_turb": self._current_wind_turb,
            "wind_sigma_ms": list(getattr(self, "_current_wind_sigma_ms", [])),
            "gps_accuracy": self._current_gps_accuracy,
            # 2026-07-29: v3 cell identity (was silently absent; callbacks.py
            # :649 info.get() therefore wrote "" to metrics.csv every episode).
            "v3_cell_label": self._current_v3_cell_label,
            "v3_rehearsal": self._current_v3_rehearsal,
            # #379: active leader-datalink transport delay (per-episode config
            # echo of record on the consumer side; 0.0 = knob off).
            "leader_tx_delay_s": float(
                os.environ.get("FD_LEADER_TX_DELAY_S", "0") or 0.0),
            # Patch A 2026-05-06: resolved per-episode runtime conditions.
            "wind_max_ms": self._current_wind_max_ms,
            "harness_uwb_enable": self._current_harness_uwb_enable,
            "reward_source_mode": self._current_reward_source_mode,
            # Patch C2 2026-05-06: paired-channel telemetry summary.
            "condition_profile": self._current_condition_profile or "",
            "paired_csv_path": (
                str(self._paired_csv_path) if self._paired_csv_path else ""
            ),
            "hallucinated_env_m3b_count": self._paired_hallucinated_env_m3b_count,
            "hallucinated_ctrl_m3b_count": self._paired_hallucinated_ctrl_m3b_count,
            "paired_env_raw_behind_samples": len(self._paired_delta_env_raw_behind),
            "paired_env_raw_behind_median_abs": (
                float(np.median(np.abs(self._paired_delta_env_raw_behind)))
                if self._paired_delta_env_raw_behind else 0.0
            ),
            "paired_env_raw_behind_p95_abs": (
                float(np.percentile(np.abs(self._paired_delta_env_raw_behind), 95))
                if self._paired_delta_env_raw_behind else 0.0
            ),
            "paired_env_raw_cross_median_abs": (
                float(np.median(np.abs(self._paired_delta_env_raw_cross)))
                if self._paired_delta_env_raw_cross else 0.0
            ),
            "paired_env_raw_cross_p95_abs": (
                float(np.percentile(np.abs(self._paired_delta_env_raw_cross), 95))
                if self._paired_delta_env_raw_cross else 0.0
            ),
            "paired_env_raw_below_median_abs": (
                float(np.median(np.abs(self._paired_delta_env_raw_below)))
                if self._paired_delta_env_raw_below else 0.0
            ),
            "paired_env_raw_below_p95_abs": (
                float(np.percentile(np.abs(self._paired_delta_env_raw_below), 95))
                if self._paired_delta_env_raw_below else 0.0
            ),
            "paired_ctrl_raw_behind_median_abs": (
                float(np.median(np.abs(self._paired_delta_ctrl_raw_behind)))
                if self._paired_delta_ctrl_raw_behind else 0.0
            ),
            "paired_ctrl_raw_behind_p95_abs": (
                float(np.percentile(np.abs(self._paired_delta_ctrl_raw_behind), 95))
                if self._paired_delta_ctrl_raw_behind else 0.0
            ),
            "paired_ctrl_raw_cross_median_abs": (
                float(np.median(np.abs(self._paired_delta_ctrl_raw_cross)))
                if self._paired_delta_ctrl_raw_cross else 0.0
            ),
            "paired_ctrl_raw_cross_p95_abs": (
                float(np.percentile(np.abs(self._paired_delta_ctrl_raw_cross), 95))
                if self._paired_delta_ctrl_raw_cross else 0.0
            ),
            "paired_ctrl_raw_below_median_abs": (
                float(np.median(np.abs(self._paired_delta_ctrl_raw_below)))
                if self._paired_delta_ctrl_raw_below else 0.0
            ),
            "paired_ctrl_raw_below_p95_abs": (
                float(np.percentile(np.abs(self._paired_delta_ctrl_raw_below), 95))
                if self._paired_delta_ctrl_raw_below else 0.0
            ),
            "behind_m": geom.behind_m if geom else 0,
            "cross_m": geom.cross_m if geom else 0,
            "below_m": geom.below_m if geom else 0,
            "range_m": geom.range_m if geom else 0,
            "in_m3b": in_m3b,
            "m3b_dwell_s": self._accumulated_m3b_dwell_s,
            "terminal_type": terminal_type,
            **self._demote_info_fields(),
            "milestone_tier": self._milestone_tier,
            "best_milestone_tier": max(self._tier_first_achieved_s.keys(), default=0),
            "min_range_m": self._min_range_m if self._min_range_m != float('inf') else 0.0,
            "time_to_M2_s": self._tier_first_achieved_s.get(1, -1.0),
            "time_to_M3a_coarse_s": self._tier_first_achieved_s.get(2, -1.0),
            "time_to_M3a_fine_s": self._tier_first_achieved_s.get(3, -1.0),
            "time_to_M3b_s": self._tier_first_achieved_s.get(4, -1.0),
            "dwell_M2_s": self._tier_dwell_s.get(1, 0.0),
            "dwell_M3a_coarse_s": self._tier_dwell_s.get(2, 0.0),
            "dwell_M3a_fine_s": self._tier_dwell_s.get(3, 0.0),
            "dwell_M3b_s": self._tier_dwell_s.get(4, 0.0),
            "max_behind_overshoot_m": self._max_behind_overshoot_m,
            "max_cross_excursion_m": self._max_cross_excursion_m,
            # T-455 telemetry (see __init__ declaration block). -1.0 sentinel
            # for the truth min-range means "no truth sample this episode",
            # deliberately distinct from a genuine small range.
            "max_above_boom_fused_m": self._max_above_boom_fused_m,
            "max_above_boom_truth_m": self._max_above_boom_truth_m,
            "min_boom_tip_range_truth_m": (
                self._min_boom_tip_range_truth_m
                if self._min_boom_tip_range_truth_m != float('inf') else -1.0),
            "guard_fires_episode": self._guard_fires_episode,
            "wake_hazard_intensity": wake_intensity,
            "episode_length": self._tick_count,
            "sub_ticks_executed": sub_tick + 1,
            "env_id": self.env_id,
            "pause_count": getattr(self, '_pause_count', 0),
            "max_pause_wall_s": getattr(self, '_max_pause_wall_s', 0.0),
            "stopped_total_wall_s": getattr(self, '_total_stopped_wall_s', 0.0),
            "sitl_sim_at_last_stop": getattr(self, '_last_sitl_sim_at_stop', 0.0),
            "sitl_sim_at_last_resume": getattr(self, '_last_sitl_sim_at_resume', 0.0),
            "stall_count": self._stall_count,
            # Path 1: FSM state from full controller reference
            "fsm_state": (
                self._ctrl_ref.get_fsm_state().name
                if self._ctrl_ref is not None else "UNKNOWN"
            ),
            # v12 Commit 7: Learned-vs-baseline metrics
            "m3b_mean_residual": (
                float(np.mean(self._m3b_residual_magnitudes))
                if self._m3b_residual_magnitudes else 0.0
            ),
            "m3a_fine_mean_residual": (
                float(np.mean(self._m3a_fine_residual_magnitudes))
                if self._m3a_fine_residual_magnitudes else 0.0
            ),
            "learned_m3b_fraction": (
                float(np.mean([r > 0.05 for r in self._m3b_residual_magnitudes]))
                if self._m3b_residual_magnitudes else 0.0
            ),
            "residual_prox_correlation": (
                float(np.corrcoef(
                    self._m3b_residual_magnitudes,
                    self._m3b_prox_deltas,
                )[0, 1])
                if len(self._m3b_residual_magnitudes) >= 5 else float('nan')
            ),
            # v19: True FlightDock contact milestones (boom-tip reference, diagnostic)
            "contact_m3a_achieved": self._contact_m3a_achieved,
            "contact_m3b_achieved": (
                max(self._best_contact_m3b_dwell_s, self._continuous_contact_m3b_dwell_s)
                >= config.CONTACT_M3B_DWELL_S
            ),
            "contact_m3b_best_dwell_s": max(
                self._best_contact_m3b_dwell_s,
                self._continuous_contact_m3b_dwell_s,
            ),
            "contact_m3b_accum_dwell_s": self._accumulated_contact_m3b_dwell_s,
            "boom_tip_min_range_m": (
                self._boom_tip_min_range_m
                if self._boom_tip_min_range_m != float('inf') else -1.0
            ),
            # C006 P4: Actual milestone achievement (authoritative definitions)
            "actual_M2_achieved": (
                max(self._best_actual_m2_dwell_s,
                    self._continuous_actual_m2_dwell_s)
                >= config.ACTUAL_M2_DWELL_S
            ),
            "actual_M2_dwell_s": max(
                self._best_actual_m2_dwell_s,
                self._continuous_actual_m2_dwell_s,
            ),
            "actual_M3a_coarse_achieved": self._contact_m3a_coarse_achieved,
            "actual_M3a_coarse_dwell_s": (
                self._continuous_contact_m3a_coarse_dwell_s
            ),
            "actual_M3a_fine_achieved": self._contact_m3a_achieved,
            "actual_M3a_fine_dwell_s": self._continuous_contact_m3a_dwell_s,
            "actual_M3b_achieved": (
                max(self._best_contact_m3b_dwell_s,
                    self._continuous_contact_m3b_dwell_s)
                >= config.CONTACT_M3B_DWELL_S
            ),
            "actual_M3b_dwell_s": max(
                self._best_contact_m3b_dwell_s,
                self._continuous_contact_m3b_dwell_s,
            ),
            "actual_min_boom_tip_range_m": (
                self._boom_tip_min_range_m
                if self._boom_tip_min_range_m != float('inf') else -1.0
            ),
            # Campaign 3c: dwell-failure decomposition
            "m3b_entry_count": self._m3b_entry_count,
            "m3b_break_count": self._m3b_break_count,
            "m3b_mean_dwell_before_break_s": (
                self._m3b_total_dwell_before_break_s / max(self._m3b_break_count, 1)
            ),
            "m3b_last_break_axis": self._m3b_last_break_axis,
            "m3b_last_break_margin_m": self._m3b_last_break_margin_m,
            "m3b_last_break_velocity_mps": self._m3b_last_break_velocity_mps,
            "m3b_breaks_behind": self._m3b_break_axis_counts["behind"],
            "m3b_breaks_cross": self._m3b_break_axis_counts["cross"],
            "m3b_breaks_below": self._m3b_break_axis_counts["below"],
            # Campaign 3h: exit penalty diagnostics
            "penalized_exit_count": self._penalized_exit_count,
            "mean_pre_exit_continuous_dwell": (
                self._pre_exit_dwell_accum / max(self._pre_exit_dwell_count, 1)
            ),
            # reward pivot diagnostics
            "approach_suppression_active": self._approach_suppression_active,
            "r_approach_suppressed_ticks": self._r_approach_suppressed_ticks,
            "behind_axis_exit_count": self._behind_axis_exit_count,
            "non_behind_exit_count": self._non_behind_exit_count,
            "r_exit_behind_total": self._r_exit_behind_total,
            "mean_ttc_at_m3b_entry": (
                float(np.mean(self._ttc_at_m3b_entries))
                if self._ttc_at_m3b_entries else 0.0
            ),
            "mean_closure_rate_delta_in_m3b": (
                float(np.mean(self._closure_rate_delta_in_m3b))
                if self._closure_rate_delta_in_m3b else 0.0
            ),
            # Overspeed penalty diagnostics
            "r_overspeed_total": self._r_overspeed_total,
            "overspeed_tick_count": self._overspeed_tick_count,
            "max_v_excess": self._max_v_excess,
            # C006 Phase 7: Action residual statistics for passthrough monitoring
            # C3X Intervention 1: action is 2-element [speed, altitude];
            # heading_residual_mean is structurally 0.0 (axis amputated)
            # but the key is preserved for downstream-consumer compatibility.
            "speed_residual_mean": (
                float(np.mean([a[0] for a in self._action_accum]))
                if self._action_accum else 0.0
            ),
            "heading_residual_mean": 0.0,
            "altitude_residual_mean": (
                float(np.mean([a[1] for a in self._action_accum]))
                if self._action_accum else 0.0
            ),
            "mean_action_l2": (
                float(np.mean([np.linalg.norm(a) for a in self._action_accum]))
                if self._action_accum else 0.0
            ),
            # Decomposed speed logging + barrier diagnostics
            "policy_speed_delta_mean": (
                self._c3o_policy_speed_delta_sum / max(self._tick_count, 1)
            ),
            "policy_speed_delta_max": self._c3o_policy_speed_delta_max,
            "bias_speed_delta_mean": (
                self._c3o_bias_speed_delta_sum / max(self._tick_count, 1)
            ),
            "bias_active_ticks": self._c3o_bias_active_ticks,
            "bias_active_while_ahead_ticks": self._c3o_bias_active_while_ahead_ticks,
            "r_barrier_inside_mean": (
                self._c3o_r_barrier_inside_sum / max(self._c3o_barrier_inside_ticks, 1)
                if self._c3o_barrier_inside_ticks > 0 else 0.0
            ),
            "r_barrier_outside_mean": (
                self._c3o_r_barrier_outside_sum / max(self._c3o_barrier_outside_ticks, 1)
                if self._c3o_barrier_outside_ticks > 0 else 0.0
            ),
            "barrier_inside_ticks": self._c3o_barrier_inside_ticks,
            "barrier_outside_ticks": self._c3o_barrier_outside_ticks,
            "ticks_to_90pct_bias": self._c3o_ticks_to_90pct_bias,
            "first_tier2_tick": self._c3o_first_tier2_tick,
            "first_bias_active_tick": self._c3o_first_bias_active_tick,
            # Tier-4 hold-phase PSD instrumentation
            "policy_speed_delta_tier4_mean": (
                float(np.mean(self._c3p_tier4_psd_values))
                if self._c3p_tier4_ticks > 0 else 0.0
            ),
            "policy_speed_delta_tier4_p10": (
                float(np.percentile(self._c3p_tier4_psd_values, 10))
                if self._c3p_tier4_ticks > 0 else 0.0
            ),
            "policy_speed_delta_tier4_p50": (
                float(np.percentile(self._c3p_tier4_psd_values, 50))
                if self._c3p_tier4_ticks > 0 else 0.0
            ),
            "policy_speed_delta_tier4_p90": (
                float(np.percentile(self._c3p_tier4_psd_values, 90))
                if self._c3p_tier4_ticks > 0 else 0.0
            ),
            "tier4_ticks": self._c3p_tier4_ticks,
            # Velocity modulation diagnostics
            "velocity_mod_mean": (
                float(np.mean(self._c3q_velocity_mod_values))
                if len(self._c3q_velocity_mod_values) > 0 else 0.0
            ),
            "velocity_mod_ticks": len(self._c3q_velocity_mod_values),
            # Braking penalty diagnostics
            "r_hold_fwd_mean": reward_components.get("r_hold_fwd_mean", 0.0),
        }

        # === Phase 5c.A.6 ENTRYPOINT_DIAG (step_end) ===
        try:
            from . import entrypoint_diag as _ed
            if _ed.is_enabled():
                _ed.emit_event(
                    "step_end",
                    path="eval",
                    tick_source="step",
                    env_id=self.env_id,
                    step_idx=self._tick_count,
                    fsm_state=getattr(self._ctrl_ref.state, "name", "")
                        if self._ctrl_ref else "",
                    **getattr(self, "_belly_clamp_diag", {}),
                )
        except Exception:
            pass

        # v7: Pause SITL until next step() or reset() resumes it.
        self._pause_sitl()

        if _FD_T0SL_PROBE:  # module-global gate (robust to partial __init__ in tests)
            _leak = self._t0sl_probe_step(obs, total_reward)
            # Durable/periodic emit: capture the counters even when the run is cut
            # off mid-episode (total_timesteps cap) — not only on leak/termination.
            if _leak or terminated or truncated or self._t0sl_steps % 500 == 0:
                logger.warning(
                    "[T0SL_PROBE] steps=%d leaks=%d max_obs=%.4g max_reward=%.4g",
                    self._t0sl_steps, self._t0sl_leaks,
                    self._t0sl_max_obs, self._t0sl_max_reward)
        return obs, total_reward, terminated, truncated, info

    def get_runtime_config_subset(self) -> dict:
        """Return worker-side config values for forkserver propagation verification.

        Called via SubprocVecEnv.env_method() from ConfigVerificationCallback.
        """
        return {
            "velocity_gate_k": config.VELOCITY_GATE_K,
            "exit_behind_multiplier": config.EXIT_BEHIND_MULTIPLIER,
            "w_barrier": config.W_BARRIER,
            "w_barrier_outside": config.W_BARRIER_OUTSIDE,
            "forward_bias_ms": config.FORWARD_BIAS_MS,
            "ema_alpha_speed": config.EMA_ALPHA_SPEED,
            "speed_slew_rate": config.SPEED_SLEW_RATE_MPS2,
            "residual_mode": config.RESIDUAL_MODE,
            "caps_actor_lambda": config.CAPS_ACTOR_LAMBDA,
            "w_hold_fwd": config.W_HOLD_FWD,
        }

    def set_reward_stage(self, stage: int) -> None:
        """Update reward computer to new curriculum stage weights.

        Called by CurriculumCallback via SubprocVecEnv.env_method() to
        propagate stage transitions into each worker process.
        """
        if self._reward_comp is not None:
            self._reward_comp.set_stage(stage)

    def set_action_scale(self, action_scale: float) -> None:
        """Update action scale (called by CurriculumCallback on stage advance)."""
        self._action_scale = action_scale

    def set_safety_range(self, stage: int) -> None:
        """Update safety range for the new curriculum stage.

        Early stages use wider safety ranges (800m) to avoid premature
        termination during exploration. Later stages tighten to production
        levels (300m) as the agent becomes competent.
        """
        self._curriculum_stage = stage
        self._safety_range_m = config.STAGE_SAFETY_RANGE.get(
            stage, config.HARD_RANGE_MAX_M
        )

    def close(self) -> None:
        """Kill all subprocesses and clean up per-worker heartbeat."""
        # T-286: the FINAL lifetime is never followed by a relaunch, so it is
        # banked here or not at all. Inert unless FD_FDM_BANK_DIR.
        _bank_if_enabled(self.env_id)
        self._reward_terms_write()   # R0 (REWARDHYG-698)
        self._kill_procs()
        # v13 Commit 12: Clean up per-worker heartbeat file
        if hasattr(self, '_heartbeat_path') and self._heartbeat_path.exists():
            self._heartbeat_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Internal: process management
    # ------------------------------------------------------------------

    def _write_finalization_sidecar(self) -> None:
        """F-E2 (2026-07-11, task #376): env-side DURABLE finalization record written
        next to the controller-written run_summary.json (an R7.1 file whose own
        tmp+os.replace has no fsync — bfc:4750-4753). Lets harvesters fail-closed:
        env_id + terminal sim time + sha256(run_summary.json) + finalized flag, written
        tmp -> flush -> fsync -> os.replace -> dir-fsync. No-op if out_dir/summary absent.
        Internally defensive — NEVER raises (the teardown SIGTERM/SIGKILL loop must run)."""
        try:
            import json as _json
            import hashlib
            out_dir = getattr(self._ctrl_ref, "_out_dir", None)
            if not out_dir:
                return
            summary_path = os.path.join(out_dir, "run_summary.json")
            if not os.path.exists(summary_path):
                return
            with open(summary_path, "rb") as f:
                raw = f.read()
            try:
                term_sim = (_json.loads(raw).get("timebase") or {}).get("sim_end_s")
            except Exception:
                term_sim = None
            record = {"env_id": self.env_id,
                      "run_summary_sha256": hashlib.sha256(raw).hexdigest(),
                      "terminal_sim_time_s": term_sim, "finalized": True}
            sidecar = os.path.join(out_dir, "run_summary_finalized.json")
            tmp = sidecar + ".tmp"
            with open(tmp, "w") as f:
                _json.dump(record, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, sidecar)
            dfd = os.open(out_dir, os.O_DIRECTORY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except Exception:
            logger.error("Finalization sidecar write failed", exc_info=True)

    def _mark_leg_inadmissible(self, reason: str) -> None:
        """F-E2 swallow-SURFACING (RCA_MEMO N-A2): drop a LEG_INADMISSIBLE marker so
        harvesters fail-closed when the pre-teardown flush did not complete. Best-effort;
        NEVER raises (called from a teardown except-block — must not block teardown)."""
        try:
            out_dir = getattr(self._ctrl_ref, "_out_dir", None)
            if not out_dir:
                return
            with open(os.path.join(out_dir, "LEG_INADMISSIBLE.txt"), "w") as f:
                f.write(f"reason={reason}\nenv_id={self.env_id}\n")
        except Exception:
            logger.error("Failed to write LEG_INADMISSIBLE marker", exc_info=True)

    def _kill_procs(self) -> None:
        """Kill SITL and leader processes by process group."""
        # T-392a: release the epoch-stamp listener before the leader dies, so
        # a retry re-binds cleanly instead of leaking one socket per attempt.
        self._estamp_close()
        # Commit C (Rev10 §Y.2): pre-teardown force-reflush for post-
        # warmup capture (Option B). Fires while emitter/fusion/success-
        # metrics objects are still alive on the controller. Defensive
        # try/except — flush exception must not block teardown
        # (otherwise process leak via skipped SIGTERM/SIGKILL loop).
        if self._ctrl_ref is not None:
            try:
                # F-E4 (2026-07-11, task #376): the prior call to the UNDEFINED method
                # self._ctrl_ref._reset_metrics_flush() raised AttributeError on EVERY
                # teardown (swallowed below), so this final-capture reflush never ran and
                # run_summary.json froze at its mid-run snapshot. Reset the idempotence
                # guard directly (bfc:2427 self._metrics_flushed) so the flush executes.
                # ENV-SIDE — sets an attribute on the controller instance; R7.1 source intact.
                self._ctrl_ref._metrics_flushed = False
                self._ctrl_ref._flush_success_metrics(reason="pre_teardown")
                self._write_finalization_sidecar()  # F-E2: env-side durable finalization
            except Exception:
                # F-E2 swallow-SURFACING (RCA_MEMO N-A2): make the failure LOUD and mark
                # the leg inadmissible so harvesters fail-closed — but NEVER block teardown
                # (the SIGTERM/SIGKILL loop below must run or SITL/FDM/leader leak; the
                # anti-process-leak rationale at 2180-2181 is preserved by design).
                logger.error("Pre-teardown flush FAILED — run_summary may be stale; "
                             "leg flagged inadmissible", exc_info=True)
                self._mark_leg_inadmissible("pre_teardown_flush_failed")
            # #391: mirror run()'s finally for the re-opened UWB delivery diag,
            # while the controller is still alive (it is reaped below). OWN
            # try/except by design — inside the flush try, a flush raise would
            # SKIP this close, and a close raise would falsely mark the leg
            # inadmissible. Flag-gated; no-op when unset.
            try:
                _close_uwb_diag_pre_teardown(self._ctrl_ref)
            except Exception:                                 # noqa: BLE001
                logger.warning("[UWB_DIAG] pre-teardown close raised — "
                               "teardown continues", exc_info=True)

        # v7: Resume stopped processes first — a SIGSTOP'd process cannot
        # handle SIGTERM until resumed (kernel queues it but won't deliver).
        self._unpause_sitl()
        # D2 (FD_CURRICULUM_V2): clear GPS degradation before kill so a leaked
        # dropout/GLTCH can't poison a reused SITL (SITL is unpaused above so it
        # can ack the verified readback). No-op in v1 / when ladder inactive.
        self._clear_gps_ladder()
        # #309 P1: self._fdm_proc is None on native (no-op); reaped here on json
        # so the FDM cannot leak across resets.
        for proc in (self._sitl_proc, self._leader_proc, self._fdm_proc):
            if proc is not None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, OSError):
                        pass
        self._sitl_proc = None
        self._leader_proc = None
        self._fdm_proc = None
        # T-29 O1: drop the cached orphan-ArduPlane PID so the next launch
        # re-resolves it (belt-and-braces; _signal_arduplane also re-verifies
        # the cmdline on every use, so a recycled PID is never signalled).
        self._arduplane_pid = None
        if self._sitl_log_file is not None:
            try:
                self._sitl_log_file.close()
            except Exception:
                logger.debug("Failed to close SITL log file", exc_info=True)
            self._sitl_log_file = None
        if self._leader_log_file is not None:
            try:
                self._leader_log_file.close()
            except Exception:
                logger.debug("Failed to close leader log file", exc_info=True)
            self._leader_log_file = None
        if self._fdm_log_file is not None:  # #309 P1
            try:
                self._fdm_log_file.close()
            except Exception:
                logger.debug("Failed to close FDM log file", exc_info=True)
            self._fdm_log_file = None
        if self._leader_rx is not None:
            try:
                mav = getattr(self._leader_rx, "mav", None)
                if mav is not None:
                    mav.close()
            except Exception:
                logger.debug("Failed to close leader MAVLink connection", exc_info=True)
            self._leader_rx = None
        if self._mav is not None:
            try:
                self._mav.close()
            except Exception:
                logger.debug("Failed to close follower MAVLink connection", exc_info=True)
            self._mav = None
        # Path 1: Release controller reference (its MAVLink is closed above)
        self._ctrl_ref = None
        self._last_baseline_cmd = None
        # Patch C2 2026-05-06: flush + close paired-channel CSV.
        self._close_paired_csv()
        self._close_flight_state()
        self._last_guided_alt_target = 0.0
        self._last_guided_alt_time = 0.0
        self._last_reposition_time = 0.0
        self._last_reposition_hdg = 0.0
        self._last_speed_send_time = 0.0
        self._last_speed_send_value = 0.0
        self._last_reposition_alt = 0.0
        self._last_alt_cmd_target = 0.0
        self._alt_err_exceed_start = None
        self._kill_orphans()

    def _kill_orphans(self) -> None:
        """Sweep orphan ArduPlane/MAVProxy processes by port pattern.

        Inside start_formation_sitl_fixed.sh, ArduPlane is backgrounded with &,
        placing it in a separate process group from the shell. os.killpg() in
        _kill_procs() kills the shell + foreground MAVProxy but ArduPlane (and
        any MAVProxy children it spawned) escape. This method uses port-specific
        pkill patterns so parallel envs don't accidentally kill each other.
        """
        base_port = config.SITL_SERIAL_PORT_BASE + self._port_offset
        patterns = [
            f"serial0=tcp:{base_port}",            # ArduPlane SITL
            f"tcp:127.0.0.1:{base_port}",          # MAVProxy connecting to SITL
            f"leader_broadcast_fixed.py.*{self._leader_port}",  # leader broadcast
        ]
        for pat in patterns:
            try:
                subprocess.run(
                    ["pkill", "-9", "-f", pat],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            except Exception:
                logger.debug("Failed to pkill orphan pattern %s", pat, exc_info=True)

    def _signal_arduplane(self, base_port: int, sig: int) -> bool:
        """T-29 O1: signal the orphan ArduPlane via a CACHED pid (fast path).

        Returns True iff the signal was delivered; False means the caller must
        fall back to `pkill` (unchanged legacy behaviour). The PID is resolved
        once per SITL launch by a pure-/proc scan and re-verified on every use,
        so a recycled PID can never be signalled.
        """
        needle = f"serial0=tcp:{base_port}"
        pid = getattr(self, "_arduplane_pid", None)
        if pid is not None:
            # Re-verify identity on EVERY use — exe link AND port — so a dead
            # or recycled PID can never be signalled.
            exe = _exe_of(pid)
            cmd = _cmdline_of(pid)
            if (not exe or os.path.basename(exe) != "arduplane"
                    or not cmd or needle not in cmd):
                pid = None
                self._arduplane_pid = None
        if pid is None:
            pid = _find_arduplane_pid(needle)
            self._arduplane_pid = pid
            # One-time witness per resolution so a run can PROVE the fast path
            # engaged. Without it a silent fallback to pkill is indistinguishable
            # from the fix working ("a witness that only prints is not a gate" —
            # this is the gate's input, checked by the F-3 bar).
            logger.info("[O1_FASTPATH] env_id=%s port_needle=%s resolved_pid=%s",
                        getattr(self, "env_id", "?"), needle,
                        pid if pid is not None else "NONE->pkill_fallback")
        if pid is None:
            return False
        try:
            os.kill(pid, sig)
            return True
        except (ProcessLookupError, OSError):
            self._arduplane_pid = None
            return False

    def _pause_sitl(self) -> None:
        """Freeze SITL + leader processes via SIGSTOP to prevent idle drift.

        When SubprocVecEnv synchronises across workers, idle envs' SITL
        processes continue advancing in real-time. SIGSTOP freezes them
        at the kernel level — zero CPU cycles, zero sim-time drift.

        Process group SIGSTOP covers bash + MAVProxy children. The
        orphan-pattern pkill covers ArduPlane which runs in its own
        process group (backgrounded with & in start_formation_sitl_fixed.sh).
        """
        # #309 P1: under JSON lockstep the SITL blocks on FDM servo I/O — there
        # is no wall-clock idle drift to freeze, and a SIGSTOP'd SITL would
        # starve the FDM's recvfrom (recv-timeout) → deadlock/episode death.
        # T-29 (2026-07-22) MEASURED the price of that no-op: with nothing
        # frozen, a sibling env's ~460 s warmup lets THIS pair free-run the
        # same span of UNCOMMANDED sim (leader recedes at 24.7 m/s; the episode
        # then dies SAFETY_RANGE on refreshed perception). FD_JSON_PAIR_FREEZE
        # freezes SITL and FDM *together*, which dissolves the #309 starvation
        # argument — a stopped FDM is not waiting on a recv — provided the FDM
        # runs with a blocking socket (enforced at __init__).
        # Scoped to the json backend on BOTH conjuncts, so the flag can never
        # alter the native path even if _fdm_proc were somehow populated there.
        _pair_freeze = (getattr(self, "_json_pair_freeze", False)
                        and getattr(self, "_backend", "native") == "json")
        if getattr(self, "_backend", "native") == "json" and not _pair_freeze:
            return
        if not config.IDLE_DRIFT_SIGSTOP_ENABLED:
            return
        if not hasattr(self, '_pause_count'):
            return  # Not initialized (e.g., mocked env in tests)

        # Guard: don't overwrite if already paused (avoids understating duration)
        if self._pause_wall_start is None:
            self._pause_wall_start = time.monotonic()
            # v11: record sim time at SIGSTOP for idle-drift correlation
            self._last_sitl_sim_at_stop = self._time_auth.now_s() if self._time_auth else 0.0

        # 1. Freeze SITL process group (bash wrapper + MAVProxy)
        if self._sitl_proc is not None:
            try:
                os.killpg(os.getpgid(self._sitl_proc.pid), signal.SIGSTOP)
            except (ProcessLookupError, OSError):
                pass

        # 2. Freeze orphan ArduPlane by port pattern
        base_port = config.SITL_SERIAL_PORT_BASE + self._port_offset
        if self._signal_arduplane(base_port, signal.SIGSTOP):
            pass  # fast path: cached PID, ~0.5 us
        else:
            try:
                subprocess.run(
                    ["pkill", "--signal", "STOP", "-f", f"serial0=tcp:{base_port}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
            except Exception:
                logger.debug("Failed to SIGSTOP orphan ArduPlane on port %d",
                             base_port, exc_info=True)

        # 2b. T-29: freeze the JSON-lockstep FDM LAST — the requester (SITL)
        # is already stopped, so the FDM is idle in recvfrom when it stops,
        # never mid-reply. Resume mirrors this (FDM first). json only; the
        # native path never populates _fdm_proc.
        if _pair_freeze and self._fdm_proc is not None:
            try:
                _fdm_pgid = os.getpgid(self._fdm_proc.pid)
                if _fdm_pgid != os.getpgid(0):  # never freeze our own group
                    os.killpg(_fdm_pgid, signal.SIGSTOP)
                else:
                    logger.error(
                        "PAIR_FREEZE: FDM pgid == trainer pgid (%d) — refusing "
                        "to SIGSTOP self; freeze skipped", _fdm_pgid)
            except (ProcessLookupError, OSError):
                pass

        # 3. Leader broadcaster is INTENTIONALLY NOT paused (C3a follow-on fix,
        # operator-authorized scope expansion 2026-05-07).
        # Leader's position model is sim-time-gated (dt = sim_time delta,
        # capped at 0.5s) so it does not drift while SITL is paused.
        # Leaving the broadcaster alive lets it continue sending at its 10 Hz
        # wall-rate cadence with the cached sim_time (which advances naturally
        # with SITL via mav_time_sync polling). Eliminates the
        # leader-broadcast-darkness windows observed in V2-attempt1 segment 14
        # (5+ s freeze segments where leader_age built to >4s, driving CTRL p95
        # bias and SAFETY_RANGE perception artifacts).
        # Pre-fix (commented out):
        #   if self._leader_proc is not None:
        #       try:
        #           os.killpg(os.getpgid(self._leader_proc.pid), signal.SIGSTOP)
        #       except (ProcessLookupError, OSError):
        #           pass

        self._pause_count += 1

    def _unpause_sitl(self) -> None:
        """Resume SITL + leader processes via SIGCONT.

        Resumes in leader-first order so the leader broadcaster is ready
        to respond before SITL sends its next time_boot_ms.
        """
        # #309 P1: no-op under JSON lockstep (nothing was SIGSTOP'd; see
        # _pause_sitl) UNLESS T-29 pair-freeze is on, in which case the FDM and
        # SITL were both stopped and must both be resumed.
        _pair_freeze = (getattr(self, "_json_pair_freeze", False)
                        and getattr(self, "_backend", "native") == "json")
        if getattr(self, "_backend", "native") == "json" and not _pair_freeze:
            return
        if not config.IDLE_DRIFT_SIGSTOP_ENABLED:
            return
        if not hasattr(self, '_unpause_count'):
            return  # Not initialized (e.g., mocked env in tests)

        # 0. T-29: resume the JSON-lockstep FDM FIRST so the responder is
        # already listening before the requester (SITL/ArduPlane) wakes and
        # sends its next servo frame. Also required before teardown: a stopped
        # process cannot handle SIGTERM, and _kill_procs() calls this method
        # immediately before reaping _fdm_proc.
        if _pair_freeze and self._fdm_proc is not None:
            try:
                _fdm_pgid = os.getpgid(self._fdm_proc.pid)
                if _fdm_pgid != os.getpgid(0):
                    os.killpg(_fdm_pgid, signal.SIGCONT)
            except (ProcessLookupError, OSError):
                pass

        # 1. Leader broadcaster was not paused (see _pause_sitl); no SIGCONT needed.
        # Pre-fix (commented out):
        #   if self._leader_proc is not None:
        #       try:
        #           os.killpg(os.getpgid(self._leader_proc.pid), signal.SIGCONT)
        #       except (ProcessLookupError, OSError):
        #           pass

        # 2. Resume orphan ArduPlane by port pattern
        base_port = config.SITL_SERIAL_PORT_BASE + self._port_offset
        if not self._signal_arduplane(base_port, signal.SIGCONT):
            try:
                subprocess.run(
                    ["pkill", "--signal", "CONT", "-f", f"serial0=tcp:{base_port}"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
            except Exception:
                logger.debug("Failed to SIGCONT orphan ArduPlane on port %d",
                             base_port, exc_info=True)

        # 3. Resume SITL process group (bash wrapper + MAVProxy)
        if self._sitl_proc is not None:
            try:
                os.killpg(os.getpgid(self._sitl_proc.pid), signal.SIGCONT)
            except (ProcessLookupError, OSError):
                pass

        self._unpause_count += 1

        # Track pause duration for diagnostics
        if self._pause_wall_start is not None:
            pause_dur = time.monotonic() - self._pause_wall_start
            self._total_stopped_wall_s += pause_dur
            if pause_dur > self._max_pause_wall_s:
                self._max_pause_wall_s = pause_dur
            self._pause_wall_start = None
            # v11: record sim time at SIGCONT for idle-drift correlation
            self._last_sitl_sim_at_resume = self._time_auth.now_s() if self._time_auth else 0.0

    def _launch_fdm(self) -> None:
        """#309 P1: spawn the JSON-lockstep external FDM (json backend only).

        Replicates the census-validated j6 FDM config (j6_s0_harness.py:107-109
        + j5a start_fdm) INLINE so the training substrate == the parity-stamped
        backend, WITHOUT importing analysis/ into src/. Bound on self._fdm_port
        (9002 + offset). IC-hold comes from the adopted default
        (FD_FDM_IC_HOLD_SEC=30.0). recv-timeout defaults to 30 s (NOT the j5a
        slice's 8 s) to tolerate the bash launcher's boot latency before SITL
        emits its first servo packet (FM1). Child dies with the worker
        (pdeathsig), reaped by _kill_procs().
        """
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        log_dir = os.path.join(repo_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        # T-286: the FDM below opens --log-jsonl with mode "w". Distil the
        # OUTGOING lifetime before it is destroyed. Inert unless FD_FDM_BANK_DIR.
        _bank_if_enabled(self.env_id, log_dir)
        self._fdm_log_file = open(os.path.join(log_dir, f"fdm_env{self.env_id}.log"), "w")
        cmd = [
            sys.executable, _FDM_PY,
            "--bind-port", str(self._fdm_port),
            "--mass", "4.5", "--alt", "306.0", "--airspeed", "24.7",
            "--recv-timeout-s", os.environ.get("FD_FDM_RECV_TIMEOUT_S", "30"),
            "--log-jsonl", os.path.join(log_dir, f"fdm_env{self.env_id}.jsonl"),
            "--thrust-calibration-mass", "2.0",
            "--air-density-model", "isa",
            "--home-alt-amsl", str(config.JSON_HOME_ALT_AMSL),
            "--sensor-airspeed-frame", "eas",
        ]
        # IC-hold: adopted harness default (32c64374). Opt-out FD_FDM_IC_HOLD_SEC="".
        ic_hold = os.environ.get("FD_FDM_IC_HOLD_SEC", "30.0").strip()
        if ic_hold:
            cmd += ["--ic-hold-sec", ic_hold]
        # #365 frame-origin lever: idempotent duplicate-frame reply (repeated servo
        # frame_count => byte-identical re-reply, no FDM clock advance) — collapses
        # the two boot-time candidate sim-time origins (#364). Opt-in; default OFF
        # (flag omitted) is byte-identical to pre-#365.
        if os.environ.get("FD_FDM_IDEMPOTENT_DUP", "0").strip() == "1":
            cmd += ["--idempotent-dup", "1"]
        # #465 curriculum-v3 wind (P0.2): appended ONLY when the FD_FDM_WIND_*
        # config block is present — absent => argv byte-identical to pre-#465.
        # The seed is the env-side per-(episode, env) word (A11), NOT the static
        # env var, so vectorized workers and successive episodes draw independent
        # gust realizations (twin legs: seed-v2 OFF + pinned FD_FDM_WIND_SEED +
        # matching (env_id, ep_idx) reproduce the same trace).
        wind_speed = os.environ.get("FD_FDM_WIND_SPEED_MS", "").strip()
        # T-229: off-ladder probe override — mirrors the leader-side site so
        # both aircraft share one air mass. Absent => byte-identical argv.
        _wov = os.environ.get("FD_EVAL_WIND_OVERRIDE_MS", "").strip()
        if _wov:
            wind_speed = f"{min(max(float(_wov), 0.0), 15.0):.3f}"
            logger.info("[WIND_OVERRIDE] FDM wind-speed-ms=%s (off-ladder probe)",
                        wind_speed)
        wind_sigma = os.environ.get("FD_FDM_WIND_SIGMA_MS", "").strip()
        if wind_speed or wind_sigma:
            cmd += ["--wind-speed-ms", wind_speed or "0"]
            wind_from = os.environ.get("FD_FDM_WIND_FROM_DEG", "").strip()
            if wind_from:
                cmd += ["--wind-from-deg", wind_from]
            if wind_sigma:
                cmd += ["--wind-sigma-ms", wind_sigma]
            wind_length = os.environ.get("FD_FDM_WIND_LENGTH_M", "").strip()
            if wind_length:
                cmd += ["--wind-length-m", wind_length]
            wind_seed = (self._episode_wind_seed if self._episode_wind_seed is not None
                         else int(os.environ.get("FD_FDM_WIND_SEED", "20260622") or "20260622"))
            cmd += ["--wind-seed", str(wind_seed)]
            # #465b ramp-in: keep wind at 0 through warmup, engage at a commanded
            # sim-time. Unset => flag omitted => FDM default 0 => byte-identical.
            wind_ramp_start = os.environ.get("FD_FDM_WIND_RAMP_START_S", "").strip()
            if wind_ramp_start:
                cmd += ["--wind-ramp-start-s", wind_ramp_start]
            wind_ramp_dur = os.environ.get("FD_FDM_WIND_RAMP_DUR_S", "").strip()
            if wind_ramp_dur:
                cmd += ["--wind-ramp-dur-s", wind_ramp_dur]
        self._fdm_proc = subprocess.Popen(
            cmd,
            stdout=self._fdm_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=pdeathsig_preexec,
        )

    def _preposition_airborne(self) -> None:
        """#309 P1: replicate the census-substrate preposition (json only).

        The skip_takeoff FSM assumes an ARMED, GUIDED, cruising vehicle — on
        the census substrate this is done by the harness BEFORE the FSM starts
        (j5a_slice_harness.py:459 preposition(), census-validated 14/14).
        Without it the vehicle is an unarmed glider: measured TAS pinned
        ~14-15 m/s, sink ~1.8 m/s, REJOIN divergence, no M2 (iterations 4-6).
        Sequence on a direct serial1 link (5762+offset; serial0=MAVProxy,
        serial2=leader): ARMING_CHECK=0 -> mode GUIDED(15) -> arm (retried to
        HEARTBEAT armed) -> DO_CHANGE_SPEED cruise 24.7 + DO_REPOSITION carrot
        ~2.2 km north @890 MSL -> hold ~35 s post-arm (transition stabilize).
        Bounded ~100 s wall; raises TimeoutError (reset() retry loop handles).

        FIX-2 (#319): FD_ENV_SIMTIME_WARMUP=1 routes to the sim-time variant
        so the dwells are measured in SIM seconds; default "0" (#322 verdict:
        fix ineffective at speedup 1.0, NOT adopted) falls through to this
        legacy wall path, preserved verbatim (SG-G1-verified rollback arm).
        """
        # FIX-2: sim-time boot variant (opt-in lever) is isolated in its own
        # method so this wall-time body stays byte-for-byte as the legacy arm.
        if self._simtime_warmup:
            return self._preposition_airborne_simtime()
        from pymavlink import mavutil as _mu
        link = _mu.mavlink_connection(
            "tcp:127.0.0.1:%d" % (5762 + self._port_offset),
            source_system=255, source_component=191)
        t0 = time.monotonic()
        settle_s, arm_window_s = 6.0, 55.0
        armed = False
        last_hb = last_act = 0.0
        try:
            while time.monotonic() < t0 + settle_s + arm_window_s + 40.0:
                m = link.recv_match(blocking=False)
                if (m is not None and m.get_type() == "HEARTBEAT"
                        and m.get_srcSystem() == 1):
                    armed = bool(m.base_mode & 128)
                now = time.monotonic()
                if now - last_hb >= 1.0:
                    link.mav.heartbeat_send(
                        _mu.mavlink.MAV_TYPE_GCS,
                        _mu.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                    last_hb = now
                if now - t0 < settle_s:
                    time.sleep(0.02)
                    continue
                if now - last_act >= 1.5:
                    last_act = now
                    if not armed:
                        link.mav.param_set_send(
                            1, 1, b"ARMING_CHECK", 0.0,
                            _mu.mavlink.MAV_PARAM_TYPE_REAL32)
                        link.mav.command_long_send(
                            1, 1, _mu.mavlink.MAV_CMD_DO_SET_MODE, 0,
                            _mu.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                            15, 0, 0, 0, 0, 0)
                        link.mav.command_long_send(
                            1, 1, _mu.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                            0, 1, 0, 0, 0, 0, 0, 0)
                    else:
                        link.mav.command_long_send(
                            1, 1, _mu.mavlink.MAV_CMD_DO_CHANGE_SPEED,
                            0, 0, 24.7, -1, 0, 0, 0, 0)
                        link.mav.command_int_send(
                            1, 1, _mu.mavlink.MAV_FRAME_GLOBAL_INT,
                            _mu.mavlink.MAV_CMD_DO_REPOSITION,
                            0, 0, -1, 0, 0, 0,
                            int((-35.363261 + 0.02) * 1e7),
                            int(149.165230 * 1e7), 890.0)
                if armed and now - t0 > settle_s + 35.0:
                    break
                time.sleep(0.02)
        finally:
            try:
                link.close()
            except Exception:
                logger.debug("preposition link close failed", exc_info=True)
        if not armed:
            raise TimeoutError(
                "json preposition: vehicle never armed "
                "(census-substrate prerequisite for skip_takeoff warmup)")
        logger.info(
            f"[env_id={self.env_id}] json preposition complete: armed+GUIDED, "
            f"cruise carrot set ({time.monotonic()-t0:.1f}s)")

    def _preposition_airborne_simtime(self) -> None:
        """FIX-2 sim-time variant of _preposition_airborne (default; #319 RCA).

        Identical MAVLink sequence to the legacy wall path, but the settle,
        arm/reposition action cadence, and post-arm hold are measured in SIM
        seconds from a LOCAL SimTimeAuthority fed by harvested follower
        time_boot_ms (ATTITUDE / SYSTEM_TIME / GLOBAL_POSITION_INT, ~3 Hz
        default on SERIAL1 — NO stream-rate change requested). This makes the
        boot a function of sim time, killing the measured warmup_wall->m2_sim
        rho=0.991 wall coupling. The local authority is isolated (no
        set_time_authority call) and used ONLY as a clock (now_s()); its
        internal metrics are never read. The settle timeline starts at the
        first follower time_boot_ms (sim_anchor); an independent WALL ceiling is
        the anti-hang backstop and the GCS heartbeat stays on wall (liveness),
        so this can never deadlock. Raises TimeoutError on no-arm (reset()
        retry loop handles).
        """
        from pymavlink import mavutil as _mu
        link = _mu.mavlink_connection(
            "tcp:127.0.0.1:%d" % (5762 + self._port_offset),
            source_system=255, source_component=191)
        wall_t0 = time.monotonic()
        settle_s, arm_window_s = 6.0, 55.0
        sta = SimTimeAuthority(drl_mode=True, verbose=False)
        sim_anchor: Optional[float] = None  # sim-time at settle start (first ready)
        armed = False
        last_hb = 0.0      # wall cadence (GCS liveness heartbeat)
        last_act = 0.0     # sim-phase cadence (arm/reposition)
        try:
            while time.monotonic() < wall_t0 + settle_s + arm_window_s + 40.0:
                m = link.recv_match(blocking=False)
                if m is not None:
                    if (m.get_type() == "HEARTBEAT"
                            and m.get_srcSystem() == 1):
                        armed = bool(m.base_mode & 128)
                    if _prep_feed_sim_clock(sta, m) and sim_anchor is None:
                        sim_anchor = sta.now_s()  # anchor on first sim stamp
                wall_now = time.monotonic()
                if wall_now - last_hb >= 1.0:
                    link.mav.heartbeat_send(
                        _mu.mavlink.MAV_TYPE_GCS,
                        _mu.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                    last_hb = wall_now
                if sim_anchor is None:
                    time.sleep(0.02)  # hold until the sim clock is live
                    continue
                phase = sta.now_s() - sim_anchor  # SIM seconds since settle start
                if phase < settle_s:
                    time.sleep(0.02)
                    continue
                if phase - last_act >= 1.5:
                    last_act = phase
                    if not armed:
                        link.mav.param_set_send(
                            1, 1, b"ARMING_CHECK", 0.0,
                            _mu.mavlink.MAV_PARAM_TYPE_REAL32)
                        link.mav.command_long_send(
                            1, 1, _mu.mavlink.MAV_CMD_DO_SET_MODE, 0,
                            _mu.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                            15, 0, 0, 0, 0, 0)
                        link.mav.command_long_send(
                            1, 1, _mu.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                            0, 1, 0, 0, 0, 0, 0, 0)
                    else:
                        link.mav.command_long_send(
                            1, 1, _mu.mavlink.MAV_CMD_DO_CHANGE_SPEED,
                            0, 0, 24.7, -1, 0, 0, 0, 0)
                        link.mav.command_int_send(
                            1, 1, _mu.mavlink.MAV_FRAME_GLOBAL_INT,
                            _mu.mavlink.MAV_CMD_DO_REPOSITION,
                            0, 0, -1, 0, 0, 0,
                            int((-35.363261 + 0.02) * 1e7),
                            int(149.165230 * 1e7), 890.0)
                if armed and phase > settle_s + 35.0:
                    break
                time.sleep(0.02)
        finally:
            try:
                link.close()
            except Exception:
                logger.debug("preposition link close failed", exc_info=True)
        if not armed:
            raise TimeoutError(
                "json preposition (sim-time): vehicle never armed "
                "(census-substrate prerequisite for skip_takeoff warmup)")
        logger.info(
            f"[env_id={self.env_id}] json preposition complete (sim-time): "
            f"armed+GUIDED, cruise carrot set "
            f"({time.monotonic()-wall_t0:.1f}s wall)")

    def _launch_sitl(self) -> None:
        """Launch ArduPilot SITL via start_formation_sitl_fixed.sh."""
        env = os.environ.copy()
        env["HARNESS_PORT_OFFSET"] = str(self._port_offset)
        if self._backend == "json":
            # #309 P1: the launcher's json branch reads FD_FDM_PORT → --sim-port-out
            # (forward-compat to P4 multi-env; == default 9002 at env_id 0).
            env["FD_FDM_PORT"] = str(self._fdm_port)
            # #309 P1 addendum: airborne IC parm (INITIAL_MODE=FBWA, census
            # substrate) appended LAST via the launcher's existing
            # FD_EXTRA_DEFAULTS hook. Append-safe: honors any operator value.
            _ic_parm = os.path.join(
                os.path.dirname(_TESTS_DIR), "config", "json_lockstep_airborne_ic.parm")
            _extra = env.get("FD_EXTRA_DEFAULTS", "").strip()
            env["FD_EXTRA_DEFAULTS"] = (_extra + "," + _ic_parm) if _extra else _ic_parm
        cmd = [
            "bash", _SITL_SCRIPT,
            "--direct",
            "--ui", self._ui_mode,
            "--speedup", str(self.speedup),
        ]
        # Log SITL script output for diagnostics (captures [DIAG] Step 1 kills)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        sitl_log_dir = os.path.join(repo_root, "logs")
        os.makedirs(sitl_log_dir, exist_ok=True)
        sitl_log = os.path.join(sitl_log_dir, f"sitl_env{self.env_id}.log")
        self._sitl_log_file = open(sitl_log, "w")
        self._sitl_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._sitl_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=pdeathsig_preexec,
        )

    def _launch_leader(
        self, wind_speed: float, wind_dir: float, wind_turb: float,
        profile: str = "racetrack", omega_deg_s: Optional[float] = None,
        weather_seed: Optional[int] = None,
    ) -> None:
        """Launch leader_broadcast_fixed.py.

        Phase E Batch 3a: ``profile`` selects the leader trajectory
        (racetrack / straight / circle). ``omega_deg_s`` sets the yaw rate
        for circle profile (radius = TAS / |omega|); ignored for other
        profiles.
        """
        leader_out = f"127.0.0.1:{self._leader_port}"
        timesync = f"udpin:127.0.0.1:{self._timesync_port}"
        sitl_tcp = f"127.0.0.1:{5763 + self._port_offset}"
        weather_port = 5763 + self._port_offset
        cmd = [
            sys.executable, "-u", _LEADER_SCRIPT,
            "--udp-out", leader_out,
            "--sitl-tcp", sitl_tcp,
            "--time-sync-conn", timesync,
            "--sim-speedup", str(self.speedup),
            "--wind-speed", str(wind_speed),
            "--wind-dir", str(wind_dir),
            "--sitl-weather-port", str(weather_port),
            "--profile", profile,
        ]
        # T-451: emit the LEGACY SIM_WIND_TURB knob only when it is actually
        # set. It is 0.0 on every V3 cell (the flown turbulence is the FDM
        # Dryden sigma), so a "--turbulence 0.0" token in the argv reads as
        # "no turbulence" on a leg flying sigma 0.20 -- it misled a report on
        # 2026-08-12. Guarded on the VALUE, not the backend: the legacy v1/v2
        # tables carry 0.1-0.5 and keep emitting it byte-identically, and
        # leader_broadcast_fixed.py's argparse default is 0.0, so omission is
        # equivalent on both backends.
        if float(wind_turb) > 0.0:
            cmd += ["--turbulence", str(wind_turb)]
        if profile == "circle" and omega_deg_s is not None:
            cmd.extend(["--omega-deg-s", str(omega_deg_s)])
        # #309 P1 addendum: census-substrate spawn geometry. json places the
        # leader so the warmup starts inside the census cell's instant-
        # LEAD_TURN ladder (calibrated t0 == census t0; see config constant —
        # the auto 9.26 km placement is far outside the validated cell).
        # Native: arg absent == byte-identical.
        if self._backend == "json":
            cmd.extend(["--start-offset", os.environ.get(
                "FD_LEADER_START_OFFSET", str(config.JSON_LEADER_START_OFFSET_M))])
            # T-383b F3: explicit --start-offset bypasses the broadcaster's
            # TEST-331 auto-hdg rule, so the south heading must be passed
            # explicitly. Inert for racetrack (PatternState-only heading).
            cmd.extend(["--hdg", os.environ.get(
                "FD_LEADER_HDG", str(config.JSON_LEADER_HDG_DEG))])
            # T-392a: additive epoch-stamp side channel. The env owns this
            # port because the leader consumer lives in the R7.1-frozen
            # controller and cannot parse a new message type.
            if os.environ.get("FD_EPOCH_STAMP", "1").strip() != "0":
                self._estamp_open()
                if getattr(self, "_estamp_sock", None) is not None:
                    cmd.extend(["--epoch-stamp-out",
                                f"127.0.0.1:{self._estamp_port()}"])
        # D1 (FD_DR_SEED_V2): reproducible leader turbulence. None (flag OFF) =>
        # arg omitted => WeatherModel(seed=None) legacy behavior (byte-identical).
        if weather_seed is not None:
            cmd.extend(["--weather-seed", str(int(weather_seed))])
        # Log leader output for diagnostics (instead of DEVNULL)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        log_dir = os.path.join(repo_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        leader_log = os.path.join(log_dir, f"leader_env{self.env_id}.log")
        self._leader_log_file = open(leader_log, "w")
        # #309 P1 NOTE: the leader engage gate (FD_ENGAGE_SIM_TIME_S) is
        # deliberately NOT armed here — sim-time gates are default-OFF for
        # representative runs (standing policy; the census docked UNGATED with
        # IC-hold), and the census cell's spawn-geometry calibration assumes
        # the leader flying from first-ready. (P1 iterations 5-6 measured a
        # LEAD_TURN stretch 51->112 s with/without the gate on the then-
        # UNARMED vehicle — the divergences those runs showed were the missing
        # arm, not the gate; see _preposition_airborne.) P2 determinism work
        # arms the gate deliberately via the parent env (inherited below).
        self._leader_proc = subprocess.Popen(
            cmd,
            stdout=self._leader_log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=pdeathsig_preexec,
            env=_leader_probe_env(os.environ, log_dir),
        )

    def _set_sim_param(self, param_name: str, value: float) -> None:
        """Set an ArduPilot SIM_* parameter on the follower SITL."""
        if self._mav is None:
            return
        try:
            from pymavlink import mavutil as _mu
            self._mav.mav.param_set_send(
                self._mav.target_system,
                self._mav.target_component,
                param_name.encode("ascii")[:16],
                value,
                _mu.mavlink.MAV_PARAM_TYPE_REAL32,
            )
        except Exception:
            logger.debug("Failed to set sim param %s=%s", param_name, value, exc_info=True)

    # ------------------------------------------------------------------
    # D2 (FD_CURRICULUM_V2): GPS degradation ladder helpers (env-side only)
    # ------------------------------------------------------------------

    def _resolve_gps_param_names(self) -> Dict[str, Any]:
        """GPS param family per FD_FW_VARIANT (legacy/mav_clean vs modern).
        Note the inverted dropout polarity: legacy ENABLE=0 drops; modern
        DISABLE=1 drops."""
        variant = os.environ.get("FD_FW_VARIANT", "legacy")
        if variant == "modern":
            return {"acc": "SIM_GPS_ACC", "gltch_x": "SIM_GPS_GLITCH_X",
                    "gltch_y": "SIM_GPS_GLITCH_Y", "dropout_param": "SIM_GPS_DISABLE",
                    "dropout_on": 1.0, "dropout_off": 0.0}
        return {"acc": "SIM_GPS1_ACC", "gltch_x": "SIM_GPS1_GLTCH_X",
                "gltch_y": "SIM_GPS1_GLTCH_Y", "dropout_param": "SIM_GPS1_ENABLE",
                "dropout_on": 0.0, "dropout_off": 1.0}

    def _set_sim_param_verified(self, param_name: str, value: float,
                                retries: int = 3, timeout_s: float = 0.5,
                                tol: float = 1e-3) -> bool:
        """Set a SIM_* param AND confirm via PARAM_VALUE readback. Returns True
        iff confirmed. For low-frequency writes (ACC, dropout edges, teardown) —
        NOT the 5 Hz GLTCH stream (intentionally fire-and-forget)."""
        if self._mav is None:
            return False
        from pymavlink import mavutil as _mu
        name16 = param_name.encode("ascii")[:16]
        want = float(value)
        for _ in range(max(1, retries)):
            try:
                self._mav.mav.param_set_send(
                    self._mav.target_system, self._mav.target_component,
                    name16, want, _mu.mavlink.MAV_PARAM_TYPE_REAL32)
            except Exception:
                logger.debug("param_set_send failed %s=%s", param_name, value, exc_info=True)
                return False
            deadline = time.time() + timeout_s
            while time.time() < deadline:
                msg = self._mav.recv_match(type="PARAM_VALUE", blocking=False)
                if msg is not None and str(msg.param_id).rstrip("\x00") == param_name[:16]:
                    if abs(float(msg.param_value) - want) <= max(tol, abs(want) * tol):
                        return True
        logger.warning("[env_id=%s] param %s=%s not confirmed (readback)",
                       self.env_id, param_name, value)
        return False

    def _apply_gps_degradation(self, gps_accuracy: float,
                               gps_dropout_enabled: bool = False,
                               gps_dropout_max: int = 0) -> None:
        """Apply per-episode GPS degradation post-warmup. v2 => ACC=CEP
        (verified) + seeded OU GLTCH walk + dropout schedule; v1 => legacy
        reported-accuracy write (byte-identical). Shared by pooled + non-pooled."""
        if not self._curriculum_v2:
            self._set_sim_param("SIM_GPS_ACCURACY", gps_accuracy)
            return
        self._set_sim_param_verified(self._gps_params["acc"], gps_accuracy)
        rng = self._episode_rng
        if rng is None:
            base = self._initial_seed if self._initial_seed is not None else 0
            rng = np.random.default_rng(np.random.SeedSequence(
                [int(base), int(self.env_id), int(self._current_ep_idx or 0), 0x6753]))
        self._gps_walk = gps_ladder.GpsOuWalk(
            cep_m=gps_accuracy, tau_s=config.GPS_OU_TAU_S,
            dt_s=1.0 / config.GPS_UPDATE_HZ, rng=rng)
        self._gps_dropout_sched = gps_ladder.build_dropout_schedule(
            enabled=bool(gps_dropout_enabled), max_events=int(gps_dropout_max),
            episode_s=config.EPISODE_DURATION_S, rng=rng)
        self._gps_dropout_active = False
        self._gps_last_apply_sim_s = -1.0

    def _apply_gps_ladder(self) -> None:
        """Push the OU GLTCH walk at ~the GPS rate + toggle dropout windows.
        Called each sub-tick when the v2 ladder is active (self._gps_walk set)."""
        if self._gps_walk is None or self._time_auth is None:
            return
        now_s = self._time_auth.now_s()
        if now_s - self._gps_last_apply_sim_s < (1.0 / config.GPS_UPDATE_HZ):
            return
        self._gps_last_apply_sim_s = now_s
        gx_deg, gy_deg = self._gps_walk.step()
        # Fire-and-forget at GPS cadence (GLTCH is re-read every GPS update).
        self._set_sim_param(self._gps_params["gltch_x"], gx_deg)
        self._set_sim_param(self._gps_params["gltch_y"], gy_deg)
        # Dropout windows (edge-triggered, verified writes).
        want_drop = any(s <= now_s < e for (s, e) in self._gps_dropout_sched)
        if want_drop and not self._gps_dropout_active:
            self._set_sim_param_verified(self._gps_params["dropout_param"],
                                         self._gps_params["dropout_on"])
            self._gps_dropout_active = True
        elif not want_drop and self._gps_dropout_active:
            self._set_sim_param_verified(self._gps_params["dropout_param"],
                                         self._gps_params["dropout_off"])
            self._gps_dropout_active = False

    def _clear_gps_ladder(self) -> None:
        """Clear GPS degradation (GLTCH=0, ACC=0, dropout off) + the schedule so
        a leaked dropout/bias can't poison a reused SITL. Verified readback when
        the SITL is alive; always resets local state. No-op in v1 / inactive."""
        sitl_alive = (self._sitl_proc is not None and self._sitl_proc.poll() is None)
        if (self._curriculum_v2 and self._gps_walk is not None
                and self._mav is not None and sitl_alive):
            try:
                self._set_sim_param_verified(self._gps_params["gltch_x"], 0.0)
                self._set_sim_param_verified(self._gps_params["gltch_y"], 0.0)
                self._set_sim_param_verified(self._gps_params["acc"], 0.0)
                self._set_sim_param_verified(self._gps_params["dropout_param"],
                                             self._gps_params["dropout_off"])
            except Exception:
                logger.debug("GPS ladder teardown-clear failed", exc_info=True)
        self._gps_walk = None
        self._gps_dropout_sched = []
        self._gps_dropout_active = False

    # ------------------------------------------------------------------
    # Internal: components
    # ------------------------------------------------------------------

    def _init_components(self, stage: int = 0) -> None:
        """Create or reset DRL-specific components.

        Components inherited from warmup (already populated by _run_warmup):
          - _geom_calc — GeometryCalculator with correct home position
          - _time_auth — SimTimeAuthority (drl_mode=True)
          - _control_laws — FormationControlLaws with correct speedup_factor
          - _leader_rx — LeaderTelemetryReceiver (kept alive for DRL phase)

        Components created fresh per episode (DRL-specific):
          - _obs_builder — fresh LP filter state
          - _action_proc — fresh per episode
          - _reward_comp — fresh dwell accumulators (stage-aware)
          - _envelope — lightweight, fresh
        """
        self._obs_builder = ObservationBuilder(
            speedup_factor=self.speedup, wind_channels=self._wind_channels)
        self._action_proc = ResidualActionProcessor()
        # Seed heading slew state from warmup handoff (prevents 180° slew on first tick)
        if self._follower_state is not None:
            self._action_proc.seed_heading(self._follower_state.heading_deg)
        else:
            logger.warning(f"[DRL env_id={self.env_id}] _follower_state is None at "
                           f"warmup handoff — heading seed skipped, first-tick slew possible")
        self._reward_comp = MRRLRewardComputer(stage=stage)
        self._envelope = EnvelopeChecker()

    # ------------------------------------------------------------------
    # B+ Step B fusion-reset trigger (controller-side via file-marker).
    # Mirrors env_method_c.py:_maybe_trigger_controller_reset (2041-2080).
    # In Method B (in-process), the controller does NOT tick autonomously
    # between reset() and step() — the first step()'s tick_once() consumes
    # the flag at top of _execute_tick(). So no ack-wait loop is needed
    # (env_method_c needed it because ctrl runs in a separate process).
    # Default OFF: env var unset → no flag written → behavior unchanged.
    # ------------------------------------------------------------------

    def _request_controller_fusion_reset(self) -> None:
        """Step B+ (2026-05-09): write controller-fusion-reset flag.

        Polled by tests/baseline_formation_controller.py:7561
        _check_and_apply_fusion_reset() at the top of _execute_tick().
        On consumption, the controller snapshots its current fusion
        position state and re-initializes the filter (velocities=0,
        covariance reflated). Targets the 7× warmup-drift mechanism
        documented in plan §"Pre-decision diagnostic — Why integration
        breaks perception" (2026-05-09).
        """
        if os.environ.get(
                "OPTION_C_RESET_CONTROLLER_FUSION_AT_EPISODE_START") != "1":
            return
        flag_path = os.environ.get(
            "OPTION_C_RESET_FLAG_PATH",
            f"/tmp/method_c_reset_request_env{self.env_id}.flag",
        )
        try:
            import json as _json
            with open(flag_path, "w") as fh:
                fh.write(_json.dumps({
                    "requested_at_wall_s": time.time(),
                    "env_id": self.env_id,
                    "trigger": "episode_start",
                    "pipeline": "method_b_env_formation_sitl",
                }) + "\n")
            logger.info(
                f"[RESET_REQUEST] env_id={self.env_id} flag={flag_path} "
                f"requested=YES")
        except Exception as e:
            logger.warning(
                f"[RESET_REQUEST] FAILED env_id={self.env_id}: "
                f"{type(e).__name__}: {e}")

    # ------------------------------------------------------------------
    # Internal: warmup via baseline controller
    # ------------------------------------------------------------------

    def _boot_screen_classify(self) -> Dict[str, Any]:
        """T-25: classify the freshly-warmed boot with the #513 union rule.

        PASSIVE sampling — pump + geometry recompute only. No tick_once(), no
        commands transmitted (tick_once in capture mode does not transmit, so a
        "ticked" window would fly UNCOMMANDED and contaminate both the sample
        and the episode's initial condition — disclosed deviation from
        T25_PLAN.md's ticked-window sketch). Over ~3 s the plane holds its last
        guided carrot; drift is cm-scale at matched cruise. m_a (age2), the only
        marker that fired in 20/20 measured episodes, is pure telemetry and
        fully refreshed by pumping. bias_med is APPROXIMATE here (the ctrl
        geometry cache refreshes only on controller ticks) and is recorded
        as such; T-32 owns the m_b disposition.
        """
        age2_ms: list = []
        bias_m: list = []
        n = 0
        for _ in range(_BOOT_SCREEN_TICKS):
            n += 1
            try:
                self._pump_mavlink()
                self._compute_geometry()
            except Exception:
                logger.debug("[BOOT_SCREEN] sample tick failed", exc_info=True)
            meta = getattr(self, "_raw_truth_meta", None)
            if (getattr(self, "_raw_align_mode", -1) == 0 and meta
                    and meta.get("age2_s") == meta.get("age2_s")):  # not NaN
                age2_ms.append(1000.0 * float(meta["age2_s"]))
                cg = getattr(self._ctrl_ref, "_geom_cache", None)
                rg = getattr(self, "_raw_truth_geom", None)
                if cg is not None and rg is not None:
                    bias_m.append(float(cg.behind_m) - float(rg.behind_m))
            time.sleep(config.TICK_DT_S / max(self.speedup, 0.5))
        usable = len(age2_ms) >= _BOOT_SCREEN_MIN_MODE0_FRAC * n
        a_med = float(np.median(age2_ms)) if age2_ms else None
        b_med = float(np.median(bias_m)) if bias_m else None
        m_a = bool(usable and a_med is not None
                   and a_med >= _BOOT_SCREEN_AGE2_THRESH_MS)
        # T-32: ABSOLUTE, mirroring tools/b2_screen.py exactly (the pin test
        # asserts constant equality; this comment pins the operator too — the
        # measured expressing-boot bias is NEGATIVE, #504).
        m_b = bool(usable and b_med is not None
                   and abs(b_med) >= _BOOT_SCREEN_BIAS_THRESH_M)
        return {
            "env_id": self.env_id, "ep_idx": self._current_ep_idx,
            "n_window": n, "n_mode0": len(age2_ms),
            "mode0_fraction": len(age2_ms) / n if n else 0.0,
            "usable": bool(usable), "age2_med_ms": a_med, "bias_med_m": b_med,
            "bias_is_approximate": True,  # ctrl cache frozen in passive window
            "m_a": m_a, "m_b": m_b, "fires": bool(m_a or m_b),
            "rerolls_so_far": self._boot_screen_rerolls,
        }

    def _boot_screen_record(self, decision: Dict[str, Any]) -> None:
        """T-25 disclosure: append the decision to B2_SCREEN_LOG.jsonl and emit
        SCREEN_ESTIMAND.txt once. The log IS the screened-population statement
        (PREREG_513 §1); a verdict quoting a screened rate without the
        all-attempts rate violates T-26 R4."""
        import json as _json  # matches the file's established local-import pattern
        logger.info("[BOOT_SCREEN] %s", _json.dumps(decision, sort_keys=True))
        out_dir = getattr(self, "_out_dir", None)
        if not out_dir:
            return
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "B2_SCREEN_LOG.jsonl"), "a",
                      buffering=1) as fh:
                fh.write(_json.dumps(decision, sort_keys=True) + "\n")
            est = os.path.join(out_dir, "SCREEN_ESTIMAND.txt")
            if not os.path.exists(est):
                with open(est, "w") as fh:
                    fh.write(
                        "FD_TRAIN_BOOT_SCREEN=1: the population this run "
                        "certifies is CLEAN-BOOT CONDITIONAL, not "
                        "unconditional. The screen's firing rule is "
                        "outcome-correlated [MEASURED, T25_PRECONDITION.md: "
                        "fired median dwell 0.25 s vs clear 15.15 s, "
                        "point-biserial r=-0.524, n=20] — discarding fired "
                        "boots removes poor performers, so screened rates are "
                        "biased UP versus all-attempts rates. Quote BOTH "
                        "(B2_SCREEN_LOG.jsonl carries every attempt).\n")
        except Exception:
            logger.debug("[BOOT_SCREEN] disclosure write failed", exc_info=True)

    # ------------------------------------------------------------------
    # P1-A item 9 (D1 rider): within-boot passive-vs-ticked align census
    # ------------------------------------------------------------------

    def _align_census_passive(self) -> None:
        """Item 9 phase 1: passive `_raw_align_mode` census on the FIRST boot.

        T-33's root cause is [HYPOTHESIS] because the 70.4 %-ticked vs
        5.6 %-passive contrast compares two corpora AND two methods. This
        census removes the confound: SAME boot, SAME window length
        (_BOOT_SCREEN_TICKS samples at TICK_DT_S cadence), passive sampling
        first (pump + geometry recompute, exactly the screen's method), then
        the first _BOOT_SCREEN_TICKS episode sub-ticks counted in step().
        Pure logging; decides T-33 D3-vs-D4. FAILS SAFE: any exception is
        swallowed, the artifact says INCONCLUSIVE, and neither reset() nor the
        episode is disturbed.
        """
        if (not getattr(self, "_align_census_enabled", False)
                or getattr(self, "_align_census_done", True)):
            return
        try:
            passive: Dict[str, int] = {}
            for _ in range(_BOOT_SCREEN_TICKS):
                try:
                    self._pump_mavlink()
                    self._compute_geometry()
                except Exception:
                    logger.debug("[ALIGN_CENSUS] passive sample failed",
                                 exc_info=True)
                mode = str(getattr(self, "_raw_align_mode", -1))
                passive[mode] = passive.get(mode, 0) + 1
                time.sleep(config.TICK_DT_S / max(self.speedup, 0.5))
            self._align_census_state = {
                "passive": passive, "ticked": {},
                "ticks_remaining": _BOOT_SCREEN_TICKS,
                "ep_idx": self._current_ep_idx,
            }
        except Exception:
            # Census must never block a boot: mark done + INCONCLUSIVE.
            logger.warning("[ALIGN_CENSUS] passive phase failed — "
                           "INCONCLUSIVE", exc_info=True)
            self._align_census_write(status="INCONCLUSIVE",
                                     reason="passive phase raised")
            self._align_census_state = None
            self._align_census_done = True

    def _align_census_tick(self) -> None:
        """Item 9 phase 2: called from step() after each geometry compute."""
        st = self._align_census_state
        if st is None:
            return
        try:
            mode = str(getattr(self, "_raw_align_mode", -1))
            st["ticked"][mode] = st["ticked"].get(mode, 0) + 1
            st["ticks_remaining"] -= 1
            if st["ticks_remaining"] <= 0:
                self._align_census_write(status="COMPLETE", reason=None)
                self._align_census_state = None
                self._align_census_done = True
        except Exception:
            logger.warning("[ALIGN_CENSUS] tick phase failed — INCONCLUSIVE",
                           exc_info=True)
            self._align_census_write(status="INCONCLUSIVE",
                                     reason="tick phase raised")
            self._align_census_state = None
            self._align_census_done = True

    def _align_census_abort_if_armed(self) -> None:
        """Re-arm guard: a reset() while the ticked window is part-filled would
        let the window span TWO boots — the exact confound item 9 exists to
        remove. Finalize INCONCLUSIVE instead. No-op when not armed."""
        if getattr(self, "_align_census_state", None) is not None:
            self._align_census_write(
                status="INCONCLUSIVE",
                reason="episode ended before ticked window filled")
            self._align_census_state = None
            self._align_census_done = True

    def _align_census_write(self, status: str, reason: Optional[str]) -> None:
        """Emit ALIGN_CENSUS.json (or log-only when out_dir is unset)."""
        import json as _json  # matches the file's established local-import pattern
        st = getattr(self, "_align_census_state", None) or {}
        n_pass = sum(st.get("passive", {}).values())
        n_tick = sum(st.get("ticked", {}).values())
        doc = {
            "item": "P1-A item 9 within-boot passive-vs-ticked align census",
            "env_id": self.env_id, "ep_idx": st.get("ep_idx"),
            "window_len": _BOOT_SCREEN_TICKS, "status": status,
            "reason": reason,
            "passive_counts_by_mode": st.get("passive", {}),
            "ticked_counts_by_mode": st.get("ticked", {}),
            "passive_mode0_frac": (st.get("passive", {}).get("0", 0) / n_pass
                                   if n_pass else None),
            "ticked_mode0_frac": (st.get("ticked", {}).get("0", 0) / n_tick
                                  if n_tick else None),
        }
        logger.info("[ALIGN_CENSUS] %s", _json.dumps(doc, sort_keys=True))
        out_dir = getattr(self, "_out_dir", None)
        if not out_dir:
            return
        try:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "ALIGN_CENSUS.json"), "w") as fh:
                _json.dump(doc, fh, indent=1, sort_keys=True)
        except Exception:
            logger.debug("[ALIGN_CENSUS] artifact write failed", exc_info=True)

    def _reward_terms_write(self) -> None:
        """R0 (REWARDHYG-698): declare which enumerated reward terms the reward
        path actually PRODUCED this lifetime, and which are enumerated-but-inert.

        The telemetry schema (``_MRRL_RAW_COMPONENTS``, callbacks._FIELDNAMES)
        still enumerates the five terms D6.1 Amendment-R (SIGNED 2026-07-18)
        TRIMMED, so they land in metrics.csv as a literal 0.000 -- indistinguishable
        from "the term ran and paid nothing". That ambiguity misread the D2a TD3
        stall analysis. The columns are NOT renamed or removed (readers and pins
        depend on the schema); this sidecar carries the meaning instead.

        DERIVED, not asserted: ``inert`` is the enumerated set minus the keys
        observed coming out of ``reward_mrrl.compute`` this lifetime. Un-trim a
        term and the declaration follows the code without being edited.
        Never raises -- teardown must run.
        """
        import json as _json  # matches the file's established local-import pattern
        try:
            seen = set(getattr(self, "_mrrl_keys_seen", ()) or ())
            enumerated = list(_MRRL_RAW_COMPONENTS)
            inert = sorted(t for t in enumerated if t not in seen)
            doc = {
                "item": "R0 reward-term declaration (REWARDHYG-698)",
                "env_id": self.env_id,
                "status": "OK" if seen else "INCONCLUSIVE",
                "reason": None if seen else "no reward tick observed this lifetime",
                "amendment_of_record": "D6.1 Amendment-R (SIGNED 2026-07-18)",
                "telemetry_enumerated": sorted(enumerated),
                "produced_by_reward": sorted(seen & set(enumerated)),
                "enumerated_but_inert": inert,
                "note": ("terms in enumerated_but_inert are written 0.000 by the "
                         "telemetry schema and were NEVER computed; a 0.000 in "
                         "those columns is structural, not a measurement"),
            }
            logger.info("[REWARD_TERMS] %s", _json.dumps(doc, sort_keys=True))
            out_dir = getattr(self, "_out_dir", None)
            if not out_dir:
                return
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "REWARD_TERMS_DECLARED.json"), "w") as fh:
                _json.dump(doc, fh, indent=1, sort_keys=True)
        except Exception:
            logger.debug("[REWARD_TERMS] declaration write failed", exc_info=True)

    # ---- T-392a: epoch-stamp side channel (observability; see module head) --
    def _estamp_port(self) -> int:
        """Dedicated UDP port, disjoint from 14550/14551/14552/14553 inside the
        same PORT_STRIDE block, so lanes never collide."""
        return 14554 + self._port_offset

    def _estamp_open(self) -> None:
        """Bind the listener. Failure leaves it None => columns stay empty and
        every other behaviour is byte-identical (an observability channel must
        never kill a leg)."""
        self._estamp_sock = None
        self._estamp_last = None
        if os.environ.get("FD_EPOCH_STAMP", "1").strip() == "0":
            return
        try:
            import socket as _s
            k = _s.socket(_s.AF_INET, _s.SOCK_DGRAM)
            k.setsockopt(_s.SOL_SOCKET, _s.SO_REUSEADDR, 1)
            k.bind(("127.0.0.1", self._estamp_port()))
            k.setblocking(False)
            self._estamp_sock = k
        except Exception:
            self._estamp_sock = None

    def _estamp_close(self) -> None:
        k = getattr(self, "_estamp_sock", None)
        if k is not None:
            try:
                k.close()
            except Exception:
                pass
        self._estamp_sock = None

    def _estamp_drain(self):
        """Return the most recent (wire_tbms, raw_latch_tbms) or None.

        Non-blocking drain-to-empty: the newest datagram wins, so a burst
        never leaves the witness reporting a stale pair."""
        k = getattr(self, "_estamp_sock", None)
        if k is None:
            return getattr(self, "_estamp_last", None)
        try:
            while True:
                try:
                    buf = k.recv(64)
                except BlockingIOError:
                    break
                except OSError:
                    break
                if not buf:
                    break
                p = buf.split()
                if len(p) == 3 and p[0] == b"ES":
                    self._estamp_last = (int(p[1]), int(p[2]))
        except Exception:
            pass
        return getattr(self, "_estamp_last", None)

    def _write_estamp_row(self, sim_time_s, episode_idx, tick_idx) -> None:
        """T-392a: one epoch-stamp witness row, opened lazily beside the paired
        CSV. Observability only — every failure is swallowed."""
        try:
            if self._paired_csv_path is None:
                return
            w = getattr(self, "_estamp_csv_writer", None)
            if w is None:
                import csv as _csv
                p = self._paired_csv_path.with_name(
                    self._paired_csv_path.name.replace("paired_channels_",
                                                       "epoch_stamp_"))
                fh = open(p, "w", newline="")
                w = _csv.DictWriter(fh, fieldnames=self._ESTAMP_CSV_FIELDS)
                w.writeheader()
                self._estamp_csv_file, self._estamp_csv_writer = fh, w
            es = self._estamp_drain()
            w.writerow({
                "sim_time_s": sim_time_s, "episode_idx": episode_idx,
                "tick_idx": tick_idx,
                "es_lead_wire": es[0] if es else "",
                "es_lead_latch": es[1] if es else "",
                "es_foll_latch": getattr(self, "_sim_state_att_tbms", -1),
            })
        except Exception:
            logger.debug("epoch_stamp.csv write failed", exc_info=True)

    def _warmup_accept_witness(self, ctrl, tag: str = "", enforce=None,
                               phase: str = "handoff") -> dict:
        """T-391: record the handoff geometry and refuse an out-of-station start.

        Reads the CONTROLLER's own perception geometry (ctrl.get_geometry())
        -- the same estimate its FSM used to declare M2. No truth field is
        touched (G4 Tier-0). Tolerances come from formation_config, the
        single source (T-386/T-389). Always records the witness dict on
        self._warmup_accept; raises TimeoutError only when enforcement is
        active AND the station check fails, so reset()'s existing retry
        loop re-boots the leg.

        ``enforce`` defaults to the module flag. The PRE-settle snapshot
        passes enforce=False: refusing there would reject a leg that the
        settle window is about to fix, which is exactly the FM7 failure.
        """
        import formation_config as _fc
        if enforce is None:
            enforce = _FD_WARMUP_ACCEPT_GATE
        geom = None
        try:
            geom = ctrl.get_geometry()
        except Exception:                      # geometry unavailable => witness it
            geom = None
        w = {"gate_enabled": _FD_WARMUP_ACCEPT_GATE, "phase": phase,
             "enforced": bool(enforce), "geom": geom is not None,
             # GATETOL-746: recorded UNCONDITIONALLY -- the refusal message
             # and the banked record both need the box even when geometry is
             # unavailable (pinned by test_fm1_gate_refuses_when_geometry_is_
             # unavailable, which caught exactly this).
             "below_tol_m": _gate_below_tol_m(_fc)}
        if geom is not None:
            w["behind_m"] = round(float(geom.behind_m), 3)
            w["cross_m"] = round(float(geom.cross_m), 3)
            w["below_m"] = round(float(geom.below_m), 3)
            w["behind_err_m"] = round(w["behind_m"] - _fc.PRECONTACT_BEHIND_M, 3)
            w["cross_err_m"] = round(w["cross_m"] - _fc.PRECONTACT_CROSS_M, 3)
            w["below_err_m"] = round(w["below_m"] - _fc.PRECONTACT_BELOW_M, 3)
            w["in_station"] = bool(
                abs(w["behind_err_m"]) <= _fc.PRECONTACT_BEHIND_OUTER_TOL_M
                and abs(w["cross_err_m"]) <= _fc.PRECONTACT_CROSS_OUTER_TOL_M
                and abs(w["below_err_m"]) <= w["below_tol_m"])
        else:
            w["in_station"] = None
        self._warmup_accept = w
        logger.info("%s T-391 warmup handoff witness: %s", tag, w)
        if enforce and w["in_station"] is not True:
            raise TimeoutError(
                f"{tag} warmup acceptance gate REFUSED the handoff: {w} "
                f"(tol behind {_fc.PRECONTACT_BEHIND_OUTER_TOL_M} / cross "
                f"{_fc.PRECONTACT_CROSS_OUTER_TOL_M} / below "
                f"{w['below_tol_m']} (GATE-LOCAL, GATETOL-746) about "
                f"{_fc.PRECONTACT_BEHIND_M}/{_fc.PRECONTACT_CROSS_M}/"
                f"{_fc.PRECONTACT_BELOW_M})")
        return w

    def _run_warmup(self, warmup_target: str = "M2") -> None:
        """Run baseline controller to M2, then keep alive for DRL tick-by-tick.

        Path 1: Uses DRLControllerRef (full FSM subclass) instead of
        WarmupController. After M2 is reached, the controller stays alive
        in command-capture mode. The DRL env drives it via tick_once().
        """
        # Rev16 §3.2 — entry assertion: stale-state guard.
        # Commit A2 — missing-safe: treat absent attribute and None identically.
        current_ep_idx = getattr(self, "_current_ep_idx", None)
        current_ep_dir = getattr(self, "_current_ep_dir", None)
        out_dir = getattr(self, "_out_dir", None)
        if current_ep_idx is None:
            raise RuntimeError(
                "_run_warmup: ep_idx not initialized — caller bypassed reset()"
            )
        if out_dir is not None and current_ep_dir is None:
            raise RuntimeError(
                "_run_warmup: _out_dir configured but ep_dir not initialized"
            )

        from .drl_controller_ref import DRLControllerRef

        tag = f"[DRL env_id={self.env_id}]"
        logger.info(f"{tag} Warmup: launching DRLControllerRef to M2...")

        # Rev16 §6 Option A — constructor-injection: both out_dir AND
        # episode_index pass THROUGH the constructor so warmup-finally
        # at baseline_formation_controller.py:8895 sees the correct
        # _out_dir / _episode_index from the start.
        ctrl = DRLControllerRef(
            follower_conn_str=f"udp:127.0.0.1:{self._follower_port}",
            speedup=self.speedup,
            leader_port=14552 + self._port_offset,
            out_dir=self._current_ep_dir,
            episode_index=self._current_ep_idx,
        )

        # v13 Commit 11: Conditional heartbeat — only touch if sim-time advances.
        # Prevents masking a deadlocked warmup (frozen sim-time but fresh heartbeat).
        _hb_stop = threading.Event()
        _hb_path = getattr(self, '_heartbeat_path', Path(config.HEARTBEAT_FILE_PATH))
        _last_sim_s = [0.0]  # Mutable container for thread access
        _lock = threading.Lock()

        def _warmup_heartbeat() -> None:
            prev_sim = 0.0
            while not _hb_stop.wait(30.0):
                with _lock:
                    current_sim = _last_sim_s[0]
                if current_sim > prev_sim:
                    _hb_path.touch(exist_ok=True)
                    prev_sim = current_sim
                # If sim-time frozen, do NOT touch — let watchdog detect stall

        def _update_sim_time(t: float) -> None:
            with _lock:
                _last_sim_s[0] = t

        _hb_thread = threading.Thread(target=_warmup_heartbeat, daemon=True)
        _hb_thread.start()
        # C005b: Unconditional heartbeat touch at warmup entry — bridges the
        # gap before sim-time starts advancing (SITL boot can take 30-60s).
        _hb_path.touch(exist_ok=True)

        # v19 Fix 1: Wall-clock timeout — run warmup in a daemon thread so
        # we can enforce a wall-clock deadline. If SITL hangs (sim-time
        # frozen), ctrl.run_warmup() blocks on MAVLink recv() forever.
        # A daemon thread + join(timeout) lets us detect and escape this.
        _timeout_s = config.WARMUP_TIMEOUT_S
        _warmup_error: list = [None]  # Mutable container for thread result

        # #309 P1 addendum: json backend spawns AIRBORNE at cruise (census
        # substrate) — warmup must start in INTERCEPT, not ground VTOL takeoff.
        # The kwarg is passed ONLY under json so the native call stays
        # LITERALLY identical to pre-addendum (mock-signature-safe; getattr
        # guards __new__-constructed test envs).
        _warmup_kw = ({"skip_takeoff": True}
                      if getattr(self, "_backend", "native") == "json" else {})

        def _warmup_worker() -> None:
            try:
                ctrl.run_warmup(
                    max_sim_seconds=self._warmup_max_sim_s,
                    sim_time_reporter=_update_sim_time,
                    **_warmup_kw,
                )
            except Exception as e:
                _warmup_error[0] = e

        _warmup_thread = threading.Thread(target=_warmup_worker, daemon=True)
        _warmup_thread.start()
        _warmup_thread.join(timeout=_timeout_s)

        _hb_stop.set()
        _hb_thread.join(timeout=5.0)
        _hb_path.touch(exist_ok=True)  # Bridge warmup→DRL handoff gap

        if _warmup_thread.is_alive():
            # Close ctrl's MAVLink socket — the daemon thread holds ctrl which
            # owns ctrl.follower.mav (not yet handed off to self._mav).
            # Without this, each timeout leaks one UDP socket.
            try:
                if hasattr(ctrl, 'follower') and hasattr(ctrl.follower, 'mav'):
                    ctrl.follower.mav.close()
            except Exception:
                pass
            raise TimeoutError(
                f"{tag} Warmup timed out after {_timeout_s:.0f}s wall-clock"
            )
        if _warmup_error[0] is not None:
            raise _warmup_error[0]

        # v7: Set AHRS_EKF_TYPE=10 (ground-truth bypass) and verify
        # FS_GCS_ENABLE=0 (disable GCS failsafe during SIGSTOP pauses).
        # Non-fatal if param_set fails — SIGSTOP eliminates EKF drift anyway.
        try:
            from pymavlink import mavutil as _mu
            mav = ctrl.follower.mav
            mav.param_set_send(
                mav.target_system, mav.target_component,
                b"AHRS_EKF_TYPE", float(config.AHRS_EKF_TYPE_DRL),
                _mu.mavlink.MAV_PARAM_TYPE_REAL32,
            )
            mav.param_set_send(
                mav.target_system, mav.target_component,
                b"FS_GCS_ENABLE", float(config.FS_GCS_ENABLE_DRL),
                _mu.mavlink.MAV_PARAM_TYPE_REAL32,
            )
        except Exception:
            logger.debug("Failed to set AHRS_EKF_TYPE/FS_GCS_ENABLE params", exc_info=True)

        logger.info(f"{tag} Warmup complete: M2 reached, controller in DRL capture mode")

        # === Keep controller alive (Path 1: full FSM as MRRL reference) ===
        self._ctrl_ref = ctrl

        # === T-391 PRE-settle snapshot (perception-side; see module head).
        # Never enforces: this is the "before" number the settle window is
        # measured against. Enforcement lives at the post-settle check at the
        # end of this method, inside reset()'s try block, so its TimeoutError
        # is caught by the existing warmup-retry loop -- no new failure path.
        self._warmup_accept_witness(ctrl, tag, enforce=False,
                                    phase="pre_settle")

        # === #391: revive the dock-state UWB delivery diag (flag-gated) ===
        # run()'s finally just closed it (warmup holds no docking state), and
        # everything the diag exists to record happens from here on. Past both
        # raise paths above => successful warmup only, once per controller.
        if _FD_UWB_FUSION_DIAG:
            _reopened = _reinit_uwb_diag_after_warmup(ctrl)
            # DISCLOSED HAZARD (cannot be fixed inside R7.1): the controller's
            # _init_uwb_fusion_diag opens mode 'w', and when
            # FD_UWB_FUSION_DIAG_DIR pins ONE path, episode N+1 truncates
            # episode N. Every gate/spawner/ab2 leg is single-episode and ab2
            # pins a PER-LEG dir, so no current caller is affected — but this
            # warns LOUDLY rather than overwriting evidence in silence.
            if _reopened and getattr(self, "_episode_index", 0) > 1 and \
                    os.environ.get("FD_UWB_FUSION_DIAG_DIR", "").strip():
                logger.warning(
                    "[UWB_DIAG] episode %d re-opened a PINNED "
                    "FD_UWB_FUSION_DIAG_DIR with mode 'w' — the previous "
                    "episode's rows are being overwritten. Pin the diag dir "
                    "PER LEG (or per episode) to keep both.",
                    getattr(self, "_episode_index", 0))

        # === Capture warmup's last commanded values for bumpless transfer ===
        self._warmup_last_speed = getattr(ctrl, '_speed_cmd_sent_last', 24.7)
        # Heading lives on FormationControlLaws, not the controller directly
        self._warmup_last_heading = getattr(ctrl.control_laws, '_last_formation_hdg_cmd', None)
        # Fallback: use follower heading from cached position
        if self._warmup_last_heading is None and ctrl.follower._last_position:
            self._warmup_last_heading = ctrl.follower._last_position.hdg / 100.0
        self._warmup_last_alt = getattr(ctrl, '_last_alt_cmd_target', None)

        # === Capture home altitude for AGL safety checks ===
        self._home_alt_msl = getattr(ctrl, "home_alt_msl", None)
        # #309 P1: in-air arming (json preposition) makes ArduPilot set HOME at
        # the ~890 MSL arm position; the controller then captures that as
        # home_alt_msl => alt_agl computes ~0 => false SAFETY_ALT on step 1
        # (measured). Override with the TRUE field elevation (the launcher's
        # --home / FDM --home-alt-amsl constant). Native path untouched.
        # (getattr: __new__-constructed test envs lack _backend.)
        if getattr(self, "_backend", "native") == "json":
            self._home_alt_msl = config.JSON_HOME_ALT_AMSL

        # === Share MAVLink connection for DRL env command sends ===
        # The raw pymavlink connection is NOT monkey-patched (only the
        # MAVConnection wrapper methods are). DRL env uses this for
        # sending final residual-modified commands.
        self._mav = ctrl.follower.mav

        # === Pin MAVLink target to follower autopilot (sys=1, comp=1) ===
        # pymavlink's mavfile.target_system is a property over self.sysid;
        # leader broadcaster (sysid=2) heartbeats during warmup latched
        # sysid to 2, sending post-HP commands to wrong target.
        # Verdict A from analysis/c4_substrate_recovery_20260424T164701Z/
        # post_h5_arduplane_receipt_ack_20260427T012131Z/.
        old_target_system = getattr(self._mav, "target_system", None)
        old_target_component = getattr(self._mav, "target_component", None)
        self._mav.target_system = 1
        self._mav.target_component = 1
        logger.warning(
            "[ENV_TARGET_PIN] old_target_system=%s old_target_component=%s "
            "new_target_system=1 new_target_component=1",
            old_target_system, old_target_component,
        )

        # === T-240: SIM_STATE rate-match (T-93 root fix, default OFF) ===
        # Sent AFTER warmup so it lands after the controller's [5/7] 10 Hz
        # request (last write wins in the per-link interval registry) and
        # BEFORE the first paired sample, so the pairing bootstrap runs
        # under deterministic co-emission (starves route-2's bad-anchor
        # latch as well as route-1's window rejection). Re-sent once at
        # 0.5 s for boot-time loss; re-asserted at each episode reset.
        self._rate_match_sent = 0
        self._rate_match_ack = None
        self._rate_match_stamps = []
        self._rate_match_logged = False
        self._rate_match_resend_due = None
        if _FD_ENV_SIMSTATE_RATE_MATCH:
            if _FD_ENV_RATE_MATCH_LEGACY_SLEEP:
                # r1 shape, verbatim — escape hatch only (see module comment).
                for _ in range(2):
                    if _send_sim_state_interval(self._mav):
                        self._rate_match_sent += 1
                    time.sleep(0.5)
            else:
                # r2: send now, defer the redundancy send to the step path so
                # the handoff never blocks (T-281; G-1 measured the harm).
                if _send_sim_state_interval(self._mav):
                    self._rate_match_sent += 1
                self._rate_match_resend_due = (time.monotonic()
                                               + _RATE_MATCH_RESEND_DELAY_S)
            logger.warning(
                "[RATE_MATCH] armed %s: SIM_STATE(108) interval request "
                "%d us, %d send(s) so far%s",
                "r1_legacy_sleep" if _FD_ENV_RATE_MATCH_LEGACY_SLEEP else "r2",
                _RATE_MATCH_INTERVAL_US, self._rate_match_sent,
                "" if _FD_ENV_RATE_MATCH_LEGACY_SLEEP
                else " (+1 deferred, non-blocking)",
            )

        # === Share time authority (already global via set_time_authority) ===
        self._time_auth = ctrl._time_auth

        # === Share geometry calculator (has correct home position from GPS fix) ===
        self._geom_calc = ctrl.geom_calc

        # === Share control laws (still needed for sim_now_s_fn rebind) ===
        self._control_laws = ctrl.control_laws
        # v6 Fix: Rebind sim_now_s_fn to DRL-phase time authority.
        self._control_laws.sim_now_s_fn = self._time_auth.now_s

        # === Seed initial baseline command from warmup's last values ===
        self._last_baseline_cmd = BaselineCommand(
            speed_ms=self._warmup_last_speed,
            heading_deg=self._warmup_last_heading or 0.0,
            altitude_m=self._warmup_last_alt or 0.0,
        )
        self._last_alt_cmd_target = self._warmup_last_alt or 0.0
        self._last_reposition_alt = self._warmup_last_alt or 0.0

        self._seed_env_state_from_ctrl(ctrl)

        # === Keep leader_rx alive for continued telemetry during DRL phase ===
        self._leader_rx = ctrl.leader_rx

        # === T-391: settle window, then the enforcing station check. Placed
        # LAST so self._mav is bound and target-pinned (:4733/:4743) -- the
        # loop must transmit, see the module-head note on capture mode.
        self._warmup_settle_wait(ctrl, tag)

    def _seed_env_state_from_ctrl(self, ctrl) -> None:
        """Copy the controller's live MAVLink caches into the env's own state.

        Extracted verbatim from the handoff block so it can also be called
        PER SETTLE SAMPLE (T-396). Without a re-seed the env channel is frozen
        at its handoff value, and a frozen channel cannot disagree -- it would
        be a rubber stamp when in-box and an un-passable gate when not.
        These caches are exactly what ``_compute_geometry()`` reads.
        """
        # === Extract last known leader telemetry from controller caches ===
        if ctrl.leader_rx and ctrl.leader_rx.last_state:
            ls = ctrl.leader_rx.last_state  # baseline LeaderState dataclass
            self._leader_state = LeaderState(
                airspeed_ms=ls.true_airspeed_ms if ls.true_airspeed_ms > 0 else 24.7,
                heading_deg=ls.heading_deg,
                alt_msl_m=ls.alt_msl,
                vd_ms=ls.vd_ms,
            )
            # Also seed position/velocity caches from leader state
            self._leader_lat = ls.lat
            self._leader_lon = ls.lon
            self._leader_vn = ls.vn_ms
            self._leader_ve = ls.ve_ms

        # === Extract last known follower state from cached MAVLink messages ===
        follower_airspeed = 24.7  # Fallback: should be at cruise by M2
        if ctrl.follower._last_vfr_hud:
            follower_airspeed = ctrl.follower._last_vfr_hud.airspeed

        if ctrl.follower._last_position:
            pos = ctrl.follower._last_position  # GLOBAL_POSITION_INT
            self._follower_lat = pos.lat / 1e7
            self._follower_lon = pos.lon / 1e7
            self._follower_vn = pos.vx / 100.0
            self._follower_ve = pos.vy / 100.0
            self._follower_state = FollowerState(
                airspeed_ms=follower_airspeed,
                alt_msl_m=pos.alt / 1000.0,
                heading_deg=pos.hdg / 100.0,
                pitch_deg=0.0,
                roll_deg=0.0,
                vd_ms=pos.vz / 100.0,
            )

        if ctrl.follower._last_attitude:
            att = ctrl.follower._last_attitude  # ATTITUDE (radians)
            if self._follower_state is not None:
                self._follower_state.pitch_deg = math.degrees(att.pitch)
                self._follower_state.roll_deg = math.degrees(att.roll)

    def _env_station_check(self, ctrl) -> Optional[dict]:
        """T-396 SECOND CHANNEL: is the ENV's own geometry inside the box?

        The controller channel and the env channel disagree with truth in
        OPPOSITE directions -- measured on the T-394 smoke, the ctrl channel
        UNDER-read the true offset by 0.86 / 1.84 m on the two failing legs
        (it thought it was on station when it was not), while the env channel
        OVER-read it by +0.40 to +1.19 m on 6/6. Requiring BOTH to be inside
        the box therefore brackets the truth between them, which neither
        channel can do alone. Both are perception-side: no truth field enters
        this check (G4 Tier-0).

        Returns None when the geometry is unavailable -- NOT a pass.
        """
        import formation_config as _fc
        try:
            self._seed_env_state_from_ctrl(ctrl)
            geom = self._compute_geometry()
        except Exception:
            geom = None
        if geom is None:
            return None
        w = {
            "env_behind_err_m": round(
                float(geom.behind_m) - _fc.PRECONTACT_BEHIND_M, 3),
            "env_cross_err_m": round(
                float(geom.cross_m) - _fc.PRECONTACT_CROSS_M, 3),
            "env_below_err_m": round(
                float(geom.below_m) - _fc.PRECONTACT_BELOW_M, 3),
        }
        # GATETOL-746: SAME gate-local box as the controller channel -- two
        # channels judged against different boxes would not bracket anything.
        w["env_below_tol_m"] = _gate_below_tol_m(_fc)
        w["env_in_station"] = bool(
            abs(w["env_behind_err_m"]) <= _fc.PRECONTACT_BEHIND_OUTER_TOL_M
            and abs(w["env_cross_err_m"]) <= _fc.PRECONTACT_CROSS_OUTER_TOL_M
            and abs(w["env_below_err_m"]) <= w["env_below_tol_m"])
        return w

    def _warmup_settle_wait(self, ctrl, tag: str = "") -> dict:
        """T-391 wait-and-verify: fly the baseline until actually on station.

        The frozen FSM declares M2 on state ENTRY, while still converging, so
        the scored window used to open ~3 m aft. This ticks the controller and
        TRANSMITS its own captured command (zero residual -- the reference-mode
        shape from step()), sampling the perception geometry until it sits
        inside the formation_config OUTER tolerances for _WARMUP_SETTLE_DWELL_N
        consecutive samples, or the sim-second budget expires.

        Outcomes: settled | timeout | departed | disabled. ``departed`` means
        the FSM left HOLD_PRECONTACT during the wait (it started docking) --
        the window opens IMMEDIATELY so nothing real goes unscored, and it is
        never a refusal. Only ``timeout`` refuses, and only when armed.
        Returns the record dict (also written per leg; see
        _write_warmup_handoff_record).
        """
        import formation_config as _fc
        pre = dict(getattr(self, "_warmup_accept", None) or {})
        rec = {
            "env_id": self.env_id,
            "ep_idx": getattr(self, "_current_ep_idx", None),
            # WARMUPREROLL disclosure: WHICH attempt and WHICH realization
            # this line was judged on (a substituted realization is on disk).
            "attempt": getattr(self, "_warmup_attempt", None),
            "weather_seed": getattr(self, "_episode_weather_seed", None),
            "wind_seed": getattr(self, "_episode_wind_seed", None),
            "reroll_enabled": os.environ.get("FD_WARMUP_REROLL", "1") != "0",
            "gate_enabled": _FD_WARMUP_ACCEPT_GATE,
            "budget_s": _WARMUP_SETTLE_BUDGET_S,
            "dwell_required": _WARMUP_SETTLE_DWELL_N,
            # GATETOL-746b: the CONFIGURATION the leg was judged under, banked
            # at top level. rec["pre"]/["post"] retain only measurements, so
            # without this the record could not prove WHICH box graded the
            # controller channel -- the env channel carried it, the ctrl one
            # did not, and an audit needs both.
            "below_tol_m": _gate_below_tol_m(_fc),
            "pre": {k: pre.get(k) for k in
                    ("behind_err_m", "cross_err_m", "below_err_m",
                     "in_station")},
        }
        if not _FD_WARMUP_ACCEPT_GATE:
            # Disarmed: no ticking, no transmitting -- byte-identical to the
            # pre-T-391 handoff. The pre-settle snapshot is still recorded.
            rec.update(outcome="disabled", wait_sim_s=0.0, n_samples=0,
                       post=rec["pre"])
            self._warmup_settle = rec
            self._write_warmup_handoff_record(rec)
            logger.info("%s T-391 settle: %s", tag, rec)
            return rec

        rec["pre_env"] = self._env_station_check(ctrl)
        t0 = self._time_auth.now_s()
        consec = 0
        n = 0
        n_in = 0
        n_both = 0
        b_min = b_max = None
        eb_min = eb_max = None
        outcome = "timeout"
        while True:
            now_s = self._time_auth.now_s()
            if now_s - t0 >= _WARMUP_SETTLE_BUDGET_S:
                outcome = "timeout"
                break
            cmd = ctrl.tick_once()
            if cmd is not None:
                self._last_baseline_cmd = cmd
            else:
                cmd = self._last_baseline_cmd
            if cmd is not None:
                # Verbatim baseline passthrough: no residual, no belly clamp,
                # no fault injection -- those are episode-scoped and the
                # episode has not started.
                self._send_commands(cmd.speed_ms, cmd.heading_deg,
                                    cmd.altitude_m)
            _st = getattr(ctrl.get_fsm_state(), "name", "")
            if _st in _SETTLE_FORWARD_STATES:
                outcome = "departed"
                break
            if _st not in _SETTLE_BAND_STATES:
                # EMERGENCY_BREAKAWAY / DOCKING_ABORT / FAIL and friends: the
                # episode's own machinery owns these. Stop waiting, never
                # refuse -- a refusal here would relabel a real safety event
                # as a warmup problem.
                outcome = "aborted"
                break
            w = self._warmup_accept_witness(ctrl, tag, enforce=False,
                                            phase="settle")
            ew = self._env_station_check(ctrl)
            n += 1
            b = w.get("behind_err_m")
            if b is not None:
                b_min = b if b_min is None else min(b_min, b)
                b_max = b if b_max is None else max(b_max, b)
            eb = (ew or {}).get("env_behind_err_m")
            if eb is not None:
                eb_min = eb if eb_min is None else min(eb_min, eb)
                eb_max = eb if eb_max is None else max(eb_max, eb)
            # DUAL-CHANNEL dwell: a sample counts only when BOTH perception
            # channels place the aircraft inside the box. An UNSAMPLED env
            # channel (None) is NOT a pass -- it breaks the dwell, so a
            # broken second channel shows up as a timeout, never as a
            # silent single-channel gate.
            if w.get("in_station") is True:
                n_in += 1
            if (w.get("in_station") is True
                    and (ew or {}).get("env_in_station") is True):
                n_both += 1
                consec += 1
            else:
                consec = 0
            if consec >= _WARMUP_SETTLE_DWELL_N:
                outcome = "settled"
                break
            time.sleep(config.TICK_DT_S / max(self.speedup, 0.5))

        wait_s = round(self._time_auth.now_s() - t0, 2)
        rec.update(outcome=outcome, wait_sim_s=wait_s, n_samples=n,
                   n_in_station=n_in, n_in_station_both=n_both,
                   consec_final=consec,
                   behind_err_min_m=b_min, behind_err_max_m=b_max,
                   env_behind_err_min_m=eb_min, env_behind_err_max_m=eb_max,
                   m2_verify_rejects=getattr(ctrl, "_m2_verify_rejects", None),
                   m2_latch_source=getattr(ctrl, "_m2_latch_source", None),
                   fsm_state_final=getattr(ctrl.get_fsm_state(), "name", ""))
        # Post-settle witness, recorded with enforce=False so the record is
        # BANKED BEFORE anything can raise: a refused attempt is precisely the
        # evidence that must survive into the next attempt's retry (its own
        # unit test caught this ordering).
        post = self._warmup_accept_witness(ctrl, tag, enforce=False,
                                           phase="post_settle")
        rec["post"] = {k: post.get(k) for k in
                       ("behind_err_m", "cross_err_m", "below_err_m",
                        "in_station")}
        rec["post_env"] = self._env_station_check(ctrl)
        pb, qb = rec["pre"].get("behind_err_m"), rec["post"].get("behind_err_m")
        if pb is not None and qb is not None:
            rec["behind_err_reduction_m"] = round(abs(pb) - abs(qb), 3)
        rec["refused"] = bool(outcome == "timeout"
                              and post.get("in_station") is not True)
        self._warmup_settle = rec
        self._write_warmup_handoff_record(rec)
        logger.info("%s T-391 settle: %s", tag, rec)
        # ENFORCE only on timeout, and only when armed. A departed leg is
        # progressing, not broken, so refusing it would burn a good boot.
        if _FD_WARMUP_ACCEPT_GATE and rec["refused"]:
            raise TimeoutError(
                f"{tag} warmup acceptance gate REFUSED the handoff after a "
                f"{wait_s:.1f}s settle window: {rec}")
        return rec

    def _t391wit3_passive_sim_state(self):
        """T391WIT3: follower TRUTH position at the
        warmup write boundary, by PASSIVE CACHE READ -- never a wire drain.

        ★ THIS IS NOT THE FIX THE RCA DRAFTED, AND THE DIFFERENCE MATTERS.
        WARMUP_TRUTH_RCA.md proposed draining SIM_STATE from ``self._mav``.
        Reading the code refuted that design before it was written: ``self._mav
        IS the controller's own connection`` (bound at :5121 from
        ``ctrl.follower.mav``), and FIX-1 2026-07-05 (#320 RCA, see
        _pump_mavlink) records that an unbounded ``recv_match`` drain on this
        SHARED link "consumed ALL wire bytes ... starving the controller's
        ATTITUDE queue" -- ~0 FSM ticks, controller stalled blind, follower
        overtook the leader in 24/24 committed legs. During warmup the
        controller is the SOLE reader and is flying the aircraft to M2, so a
        drain there aims the #320 failure mode at the worst possible phase.

        The passive read consumes nothing: ``_latest_new_sim_state`` reads
        pymavlink's last-message cache, which the controller's own pumps keep
        fresh on the shared link -- which is also why this can work at all
        during warmup, when the env itself is not stepping.

        ``last_stamp=None`` is deliberate. That argument exists to stop the
        stepping pump RE-COUNTING a sample it already consumed; this is a
        witness, not a counter, so an already-observed sample is still valid
        evidence and must not be suppressed as a duplicate.

        Writes NOTHING to self (no _sim_state_* field, no wire stamp, no
        align2 counter -- the truth seam is left byte-identical) and never
        raises. Returns (lat, lon, alt_m) or None.
        """
        try:
            mav = getattr(self, "_mav", None)
            if mav is None:
                return None
            parsed = _latest_new_sim_state(mav, None)
            if parsed is None:
                return None
            lat, lon, alt = parsed[0], parsed[1], parsed[2]
            if lat is None or lon is None or alt is None:
                return None
            return float(lat), float(lon), float(alt)
        except Exception:
            return None

    def _t391wit2_compute_truth(self):
        """T391WIT2: on-demand truth at the write
        boundary. The stepping latch `_raw_truth_geom` is populated only by
        the paired capture DURING stepping (:6053ff) and reset per episode, so
        it is STRUCTURALLY None when the warmup handoff record writes --
        T391SMOKE-736 measured truth_available=false on 6/6 armed legs.
        Recompute here with the SAME projection the paired capture uses, from
        the SIM_STATE truth the telemetry pump keeps fresh through warmup.
        The leader position is the RAW latest sample (no epoch alignment at
        this boundary); the caller labels provenance so the audit can price
        the residual leader-message age instead of mistaking it for skew.
        Observability only: returns None on ANY missing input, never raises.
        """
        try:
            lat = getattr(self, "_sim_state_lat", None)
            lon = getattr(self, "_sim_state_lon", None)
            alt = getattr(self, "_sim_state_alt_m", None)
            self._t391wit3_pos_source = "stepping_pump"
            if lat is None or lon is None or alt is None:
                # T391WIT3: during warmup the env is not stepping, so
                # _pump_mavlink has NEVER run and these are structurally None
                # (WARMUP_TRUTH_RCA.md: the single missing input). Passive
                # cache read -- consumes no wire bytes, see #320 above.
                _pass = getattr(self, "_t391wit3_passive_sim_state", None)
                _p = _pass() if _pass is not None else None
                if _p is not None:
                    lat, lon, alt = _p
                    self._t391wit3_pos_source = "passive_cache_t391wit3"
            ls = getattr(self, "_leader_state", None)
            fs = getattr(self, "_follower_state", None)
            llat = getattr(self, "_leader_lat", None)
            llon = getattr(self, "_leader_lon", None)
            if None in (lat, lon, alt, llat, llon) or ls is None or fs is None:
                return None
            import formation_config as fc
            return self._project_to_boom_frame(
                foll_lat=lat, foll_lon=lon, foll_alt=alt,
                foll_vn=getattr(self, "_follower_vn", 0.0),
                foll_ve=getattr(self, "_follower_ve", 0.0),
                foll_vd=getattr(fs, "vd_ms", 0.0),
                lead_lat=llat, lead_lon=llon,
                lead_alt=ls.alt_msl_m,
                lead_vn=getattr(self, "_leader_vn", 0.0),
                lead_ve=getattr(self, "_leader_ve", 0.0),
                lead_vd=getattr(ls, "vd_ms", 0.0),
                lead_heading_deg=ls.heading_deg,
                boom_length_m=fc.BOOM_TIP_NOMINAL_BEHIND_M,
                source="RAW_TRUTH_DIRECT",
            )
        except Exception:
            return None

    def _t391wit_truth_fields(self) -> dict:
        """T391WIT: same-tick truth geometry
        for the handoff RECORD. Returns a dict of `truth_*` keys, never raises.

        WHY IT LIVES HERE AND NOT IN THE GATE. T391REVAL-730 needed truth at
        the handoff tick; my first implementation read it inside the
        acceptance-gate method, and `test_fm3_gate_reads_perception_only_no_
        truth_field` correctly FAILED it -- that detector forbids any truth
        reference in the deciding method as a STRUCTURAL Tier-0 guard, which is
        far stronger than reasoning about which fields the decision happens to
        use. So the gate stays byte-for-byte truth-free and the witness is
        attached at the write boundary, after every decision has been made.
        The record is observability only; nothing reads it back.
        """
        import formation_config as _fc
        # truth_available established FIRST: even a degenerate partial-init
        # object (tests build these with SimpleNamespace/__new__) must get an
        # explicit absence record, never a missing key (T391WIT detectors).
        out: dict = {"truth_available": False}
        try:
            tg = getattr(self, "_raw_truth_geom", None)
            out["truth_align_mode"] = int(getattr(self, "_raw_align_mode", -1))
            if tg is not None:
                out["truth_source"] = "stepping_latch"
            else:
                # T391WIT2: the latch is structurally empty at handoff time
                # (populated only during stepping) -- compute on demand.
                _compute = getattr(self, "_t391wit2_compute_truth", None)
                tg = _compute() if _compute is not None else None
                if tg is not None:
                    out["truth_source"] = "computed_at_write_unaligned"
                    # T391WIT3: WHICH follower-truth source fed that compute.
                    # The audit must never silently treat a passive-cache
                    # sample as if the stepping pump had produced it.
                    out["truth_pos_source"] = str(getattr(
                        self, "_t391wit3_pos_source", "unknown"))
            out["truth_available"] = tg is not None
            if tg is None:
                return out
            out["truth_behind_m"] = round(float(tg.behind_m), 3)
            out["truth_cross_m"] = round(float(tg.cross_m), 3)
            out["truth_below_m"] = round(float(tg.below_m), 3)
            out["truth_behind_err_m"] = round(
                out["truth_behind_m"] - _fc.PRECONTACT_BEHIND_M, 3)
            out["truth_cross_err_m"] = round(
                out["truth_cross_m"] - _fc.PRECONTACT_CROSS_M, 3)
            out["truth_below_err_m"] = round(
                out["truth_below_m"] - _fc.PRECONTACT_BELOW_M, 3)
            out["truth_in_station"] = bool(
                abs(out["truth_behind_err_m"]) <= _fc.PRECONTACT_BEHIND_OUTER_TOL_M
                and abs(out["truth_cross_err_m"]) <= _fc.PRECONTACT_CROSS_OUTER_TOL_M
                and abs(out["truth_below_err_m"]) <= _fc.PRECONTACT_BELOW_OUTER_TOL_M)
        except Exception as exc:
            out["truth_error"] = repr(exc)[:120]
        return out

    def _write_warmup_handoff_record(self, rec: dict) -> None:
        """T-391: append one JSON line per warmup ATTEMPT to the episode dir.

        One line per attempt (not per episode) on purpose: a refused attempt
        followed by a settled one is exactly the history that has to survive,
        and a retry must never overwrite the evidence for the attempt that
        caused it. Observability only -- every failure is swallowed.

        T391WIT: the same-tick truth fields are merged HERE, at the write
        boundary, so the acceptance gate itself never touches truth (G4).
        """
        try:
            try:
                _t = self._t391wit_truth_fields()
                if _t:
                    rec = dict(rec)
                    rec.update(_t)
                    pin = rec.get("in_station")
                    if pin is not None and "truth_in_station" in _t:
                        rec["truth_percep_agree"] = bool(
                            _t["truth_in_station"] == pin)
            except Exception:
                logger.debug("T391WIT truth merge failed", exc_info=True)
            d = getattr(self, "_current_ep_dir", None) or getattr(
                self, "_out_dir", None)
            if d is None:
                return
            import json as _json  # file-established local-import pattern
            p = Path(d) / f"warmup_handoff_env{self.env_id}.jsonl"
            with open(p, "a") as fh:
                fh.write(_json.dumps(rec, default=str) + "\n")
        except Exception:
            logger.debug("warmup_handoff.jsonl write failed", exc_info=True)

    def _stash_residual_for_flight_state(self, policy_action, source):
        """TELEM-766-A: keep policy-vs-applied available to the flight record."""
        self._fs_policy_action = policy_action
        self._fs_res_source = source

    def _write_residual_action_row(
            self, policy_action, applied_action, source: str) -> None:
        """RESLOG-749: append one JSON line per step() call — the action the
        policy commanded and the action actually applied after overrides.

        SEAM OF RECORD (RESLOGFIX-786, measured 2026-08-21): `applied_action`
        here is the action AFTER the override chain (inject_zero / inject_const
        / respid2) and BEFORE the residual processor — i.e. PRE-CRRL, PRE-clamp.
        It witnesses overrides, nothing else: applied == policy means "no
        override fired", NOT "CRRL let it through". The POST-CRRL, post-clamp
        pair is the flight-state record's res_policy_* / res_applied_* columns
        (_fs_res_dict, fed by _prev_applied_residual). An authority ratio read
        from THIS file is wrong by construction; tests/unit/drl/
        test_reslog_seam_786.py pins that the two records DISAGREE on a
        CRRL-clipped tick.

        residual actions MUST be reviewable
        for every test/eval (mandatory). "Training covered for free" was the
        original claim and is FALSE as implemented: the write is gated by an
        output directory (below) that train.py never configures, so the
        training path no-ops here silently. The training-path residual record
        of record is the flight-state recorder, which is independent of
        _out_dir. Observability only — follows the
        _write_warmup_handoff_record pattern exactly: every failure is
        swallowed, the control path never depends on this write. The NEED was
        measured, not hypothetical: the NEVERARRIVED-748 zero-residual arm had
        to witness its own intervention through a log-text marker because the
        actions themselves were nowhere on disk.

        Kill-switch: FD_RESIDUAL_ACTION_LOG=0 (default ON — the operator made
        eval coverage mandatory, so absence must be a choice, never a default).
        """
        try:
            if os.environ.get(
                    "FD_RESIDUAL_ACTION_LOG", "1").strip() == "0":
                return
            d = getattr(self, "_current_ep_dir", None) or getattr(
                self, "_out_dir", None)
            if d is None:
                return
            import json as _json  # file-established local-import pattern
            rec = {
                "tick": int(getattr(self, "_tick_count", -1)),
                "sim_time_s": getattr(
                    getattr(self, "_ctrl_ref", None), "_t_sim_s", None),
                "policy_action": [
                    float(x) for x in np.asarray(policy_action).ravel()],
                "applied_action": [
                    float(x) for x in np.asarray(applied_action).ravel()],
                "source": source,
            }
            p = Path(d) / f"residual_actions_env{self.env_id}.jsonl"
            with open(p, "a") as fh:
                fh.write(_json.dumps(rec, default=str) + "\n")
        except Exception:
            logger.debug("residual_actions.jsonl write failed", exc_info=True)

    # ------------------------------------------------------------------
    # Internal: telemetry
    # ------------------------------------------------------------------

    def _pump_mavlink(self) -> None:
        """Update follower and leader telemetry from controller caches.

        Path 1: The DRLControllerRef is the sole MAVLink reader. This method
        reads from the controller's cached messages (updated by tick_once())
        and the leader_rx (updated by the controller's _pump_mavlink_messages).
        """
        if self._ctrl_ref is None:
            return

        # Patch C2 2026-05-06: SIM_STATE for RAW_TRUTH follower position.
        # PRIMARY: lat_int / lon_int int32 / 1e7 (ardupilotmega canonical).
        # Float lat / lon fields are KNOWN BROKEN under SITL
        # (project_option_c_phase36a) and must NOT be used.
        # FIX-1 2026-07-05 (#320 RCA): the legacy unbounded recv_match drain
        # (else-branch below) consumed ALL wire bytes on the SHARED MAVLink
        # connection during episodes, starving the controller's ATTITUDE
        # queue — tick_once executed ~0 FSM ticks (drain_n=0 on 299/300
        # calls), the controller stalled blind, and the follower overtook
        # the leader in 24/24 committed legs (blind flybys at 5.77-9.48 m).
        # Default path is now a PASSIVE read of the messages cache (the
        # F1-alt-1 pattern, same as the GPI read below); cache freshness
        # comes from the controller's own tick_once pumps
        # (drl_controller_ref.py:249,256), which run every sub-tick right
        # after this method — env truth channels lag the wire by <= 1
        # sub-tick. FD_ENV_SIMSTATE_PASSIVE=0 restores the legacy drain
        # verbatim (rollback arm).
        if self._mav is not None and self._simstate_passive:
            parsed = _latest_new_sim_state(
                self._mav, self._sim_state_wire_stamp)
            if parsed is not None:
                # #475 R4: absent adjacency stamp => kinematic tick-snap
                # synthesis vs the last ACCEPTED anchor (the position payload
                # is the clock; see _simstate_snap_stamp). Never overrides a
                # present stamp; a decline falls through to the legacy
                # unstamped path (align2 mode-1 hold, exactly as before).
                # FD_ENV_ALIGN2_STAMP_SYNTH=0 => R3-exact behavior.
                _synth = False
                if (_FD_ENV_ALIGN2_STAMP_SYNTH and parsed[4] is not None
                        and parsed[4] <= 0):
                    _st = _simstate_snap_stamp(
                        self._sim_state_lat, self._sim_state_lon,
                        self._sim_state_att_tbms,
                        parsed[0], parsed[1], parsed[5], parsed[6])
                    if _st is not None:
                        parsed = (parsed[0], parsed[1], parsed[2], parsed[3],
                                  _st, parsed[5], parsed[6], parsed[7])
                        _synth = True
                        self._r4_synth_accepts += 1
                    else:
                        self._r4_synth_declines += 1
                # #475 R3: pairing-consistency acceptance vs the last
                # ACCEPTED pair (see _simstate_pair_implausible). Rejected
                # candidates are marked observed (wire stamp advances) but
                # never enter the follower truth sample; 40-consec escape
                # re-locks so a pathological stream cannot starve the
                # mirror. Same kill-switch as R1/R2.
                _rej = False
                if _FD_ENV_ALIGN2_COHERENT:
                    _rej, _ = _simstate_pair_implausible(
                        self._sim_state_lat, self._sim_state_lon,
                        self._sim_state_att_tbms,
                        parsed[0], parsed[1], parsed[4],
                        parsed[5], parsed[6])
                # #475 R5: a PRESENT stamp that fails
                # pairing-consistency is a LYING CLOCK LABEL on trustworthy
                # position truth — re-stamp it via the same kinematic snap
                # instead of holding (SG-2 run 3: refusals drove 17.8-28.5%
                # holds; counterfactual re-stamp => 0 jumps on 6/6 legs).
                # Positions inconsistent with EVERY tick hypothesis still
                # reject => hold (genuine anomalies keep their guard).
                if _rej and _FD_ENV_ALIGN2_STAMP_SYNTH:
                    _st = _simstate_snap_stamp(
                        self._sim_state_lat, self._sim_state_lon,
                        self._sim_state_att_tbms,
                        parsed[0], parsed[1], parsed[5], parsed[6])
                    if _st is not None:
                        parsed = (parsed[0], parsed[1], parsed[2], parsed[3],
                                  _st, parsed[5], parsed[6], parsed[7])
                        _synth = True
                        _rej = False
                        self._r5_restamps += 1
                if _rej and self._r3_consec_rejects < 40:
                    self._r3_pair_rejects += 1
                    self._r3_consec_rejects += 1
                    self._sim_state_wire_stamp = parsed[3]
                else:
                    if _rej:
                        self._r3_relocks += 1
                    self._r3_consec_rejects = 0
                    (self._sim_state_lat, self._sim_state_lon,
                     self._sim_state_alt_m, self._sim_state_wire_stamp,
                     self._sim_state_att_tbms, self._sim_state_vn,
                     self._sim_state_ve, self._sim_state_vd) = parsed
                    self._sim_state_att_synth = _synth  # #475 R4 provenance
                    self._sim_state_last_t = (
                        self._time_auth.now_s() if self._time_auth else None
                    )
                    # T-240 witness (logging-only, one line per boot): first
                    # 10 committed 108 inter-arrivals + the 511 ACK. getattr
                    # default True => silently inert when init never armed
                    # the witness (e.g. __new__-constructed test envs).
                    if (_FD_ENV_SIMSTATE_RATE_MATCH
                            and not getattr(self, "_rate_match_logged", True)):
                        if self._rate_match_ack is None:
                            _ack = self._mav.messages.get('COMMAND_ACK')
                            if (_ack is not None
                                    and getattr(_ack, 'command', -1) == 511):
                                self._rate_match_ack = getattr(
                                    _ack, 'result', -1)
                        self._rate_match_stamps.append(float(parsed[3]))
                        if (len(self._rate_match_stamps)
                                >= _RATE_MATCH_WITNESS_N):
                            _iv = [round((_b - _a) * 1000.0, 1)
                                   for _a, _b in zip(
                                       self._rate_match_stamps,
                                       self._rate_match_stamps[1:])]
                            logger.warning(
                                "[RATE_MATCH] ack=%s intervals_ms=%s",
                                self._rate_match_ack, _iv)
                            self._rate_match_logged = True
        elif self._mav is not None:
            # Legacy unbounded drain — byte-for-byte pre-FIX-1 (rollback
            # arm). F1-alt-1 (2026-05-09): placed at TOP of _pump_mavlink so
            # that the side-effect of recv_match (parsing wire bytes into
            # self._mav.messages dict for ALL types, including
            # GLOBAL_POSITION_INT) refreshes the cache that the GPI passive
            # read below consumes.
            try:
                while True:
                    msg = self._mav.recv_match(
                        type='SIM_STATE', blocking=False)
                    if msg is None:
                        break
                    lat_int = getattr(msg, 'lat_int', None)
                    lon_int = getattr(msg, 'lon_int', None)
                    if lat_int is not None and lon_int is not None:
                        self._sim_state_lat = lat_int / 1e7
                        self._sim_state_lon = lon_int / 1e7
                    else:
                        # Float fallback (known unreliable; last resort).
                        self._sim_state_lat = float(getattr(msg, 'lat', 0.0))
                        self._sim_state_lon = float(getattr(msg, 'lon', 0.0))
                    self._sim_state_alt_m = float(getattr(msg, 'alt', 0.0))
                    self._sim_state_last_t = (
                        self._time_auth.now_s() if self._time_auth else None
                    )
            except Exception:
                # SIM_STATE drain is observability-only; never block telemetry.
                pass

        # F1-alt-1 (2026-05-09, revised): read latest GLOBAL_POSITION_INT
        # from pymavlink's internal messages cache (passive — no wire-byte
        # consumption). Cache freshness: in the FIX-1 default passive mode
        # the controller's tick_once pumps refresh it every sub-tick; in
        # the legacy rollback arm (FD_ENV_SIMSTATE_PASSIVE=0) the SIM_STATE
        # drain above refreshes it, its recv_match calls parsing ALL
        # message types side-effect-into self._mav.messages. Initial
        # F1-alt-1 implementation drained GPI via its own recv_match loop,
        # which starved the controller's _pump_mavlink_messages
        # (tests/baseline_formation_controller.py:2071-2078) of GPI bytes
        # — the controller went blind to follower position and the
        # follower flew 800 m past the leader at sim=77 s (SAFETY_RANGE
        # termination); FIX-1 (#320) later measured the SAME starvation
        # class from the SIM_STATE drain itself. Passive cache-read keeps
        # the controller's wire-byte feed intact while still giving env's
        # paired-row writer per-sub-tick GPI freshness. Falls back to the
        # controller's cached _last_position if the messages dict is
        # empty at warmup transition.
        env_gpi = None
        if self._mav is not None:
            try:
                env_gpi = self._mav.messages.get('GLOBAL_POSITION_INT', None)
            except Exception:
                env_gpi = None

        # --- Follower telemetry: prefer env's passive cache read; fall back to
        #     controller's _last_position if the cache is empty.
        pos = env_gpi if env_gpi is not None else self._ctrl_ref.follower._last_position
        if pos is not None:
            lat = pos.lat / 1e7
            lon = pos.lon / 1e7
            alt_msl = pos.alt / 1000.0
            hdg = pos.hdg / 100.0
            vn = pos.vx / 100.0
            ve = pos.vy / 100.0
            vd = pos.vz / 100.0

            self._follower_lat = lat
            self._follower_lon = lon
            self._follower_vn = vn
            self._follower_ve = ve
            if self._follower_state is None:
                self._follower_state = FollowerState(
                    airspeed_ms=24.7, alt_msl_m=alt_msl,
                    heading_deg=hdg, pitch_deg=0.0, roll_deg=0.0, vd_ms=vd,
                )
            else:
                self._follower_state.alt_msl_m = alt_msl
                self._follower_state.heading_deg = hdg
                self._follower_state.vd_ms = vd

        # TELEMFIX-771: note arrivals BEFORE the values are consumed, and note
        # them unconditionally -- a channel whose object did not change is
        # exactly the stale case the guards must be able to see.
        self._telem_note("pos", pos)
        self._telem_note("vfr", self._ctrl_ref.follower._last_vfr_hud)
        self._telem_note("att", self._ctrl_ref.follower._last_attitude)

        vfr = self._ctrl_ref.follower._last_vfr_hud
        if vfr is not None and self._follower_state is not None:
            self._follower_state.airspeed_ms = vfr.airspeed
            self._follower_state.throttle_pct = float(vfr.throttle)

        att = self._ctrl_ref.follower._last_attitude
        if att is not None and self._follower_state is not None:
            self._follower_state.pitch_deg = math.degrees(att.pitch)
            self._follower_state.roll_deg = math.degrees(att.roll)

        # --- Leader telemetry from leader_rx (has its own socket) ---
        if self._ctrl_ref.leader_rx and self._ctrl_ref.leader_rx.last_state:
            ls = self._ctrl_ref.leader_rx.last_state
            self._leader_lat = ls.lat
            self._leader_lon = ls.lon
            self._leader_vn = ls.vn_ms
            self._leader_ve = ls.ve_ms
            if self._leader_state is None:
                self._leader_state = LeaderState(
                    airspeed_ms=ls.true_airspeed_ms if ls.true_airspeed_ms > 0 else 24.7,
                    heading_deg=ls.heading_deg,
                    alt_msl_m=ls.alt_msl,
                    vd_ms=ls.vd_ms,
                )
            else:
                self._leader_state.heading_deg = ls.heading_deg
                self._leader_state.alt_msl_m = ls.alt_msl
                self._leader_state.vd_ms = ls.vd_ms
                if ls.true_airspeed_ms > 0:
                    self._leader_state.airspeed_ms = ls.true_airspeed_ms

    def _project_to_boom_frame(
        self,
        foll_lat: float,
        foll_lon: float,
        foll_alt: float,
        foll_vn: float,
        foll_ve: float,
        foll_vd: float,
        lead_lat: float,
        lead_lon: float,
        lead_alt: float,
        lead_vn: float,
        lead_ve: float,
        lead_vd: float,
        lead_heading_deg: float,
        boom_length_m: float,
        source: str,
        boom_tip_offset_m: float = fc.BOOM_TIP_OFFSET_M,
    ) -> Optional[RelativeGeometry]:
        # F1-alt 2026-05-09: replicates tests/geometry.py:GeometryCalculator
        # .compute_relative()'s NED + boom-offset + body-rotation math
        # (lines 763-799), but does NOT touch self._geom_calc.fusion. Used
        # to eliminate the env-side fusion call-order mutation confirmed in
        # analysis/p2_fusion_call_order_20260509T050603Z/P2_VERDICT.md.
        if self._geom_calc is None:
            return None
        foll_n, foll_e, foll_d = self._geom_calc.latlon_to_ned(
            foll_lat, foll_lon, foll_alt)
        lead_n, lead_e, lead_d = self._geom_calc.latlon_to_ned(
            lead_lat, lead_lon, lead_alt)
        hdg_rad = math.radians(lead_heading_deg)
        boom_offset_n = -boom_length_m * math.cos(hdg_rad)
        boom_offset_e = -boom_length_m * math.sin(hdg_rad)
        boom_n = lead_n + boom_offset_n
        boom_e = lead_e + boom_offset_e
        boom_d = lead_d + float(boom_tip_offset_m)
        rel_n = foll_n - boom_n
        rel_e = foll_e - boom_e
        rel_d = foll_d - boom_d
        cos_hdg = math.cos(hdg_rad)
        sin_hdg = math.sin(hdg_rad)
        body_x = rel_n * cos_hdg + rel_e * sin_hdg
        body_y = -rel_n * sin_hdg + rel_e * cos_hdg
        behind_m = -body_x
        cross_m = body_y
        below_m = rel_d
        range_m = math.sqrt(
            behind_m * behind_m + cross_m * cross_m + below_m * below_m
        )
        rel_vn = foll_vn - lead_vn
        rel_ve = foll_ve - lead_ve
        rel_vd = foll_vd - lead_vd
        vel_behind = -(rel_vn * cos_hdg + rel_ve * sin_hdg)
        vel_cross = -rel_vn * sin_hdg + rel_ve * cos_hdg
        vel_below = rel_vd
        target_aspect_deg = math.degrees(math.atan2(cross_m, behind_m))
        return RelativeGeometry(
            behind_m=behind_m,
            cross_m=cross_m,
            below_m=below_m,
            range_m=range_m,
            target_aspect_deg=target_aspect_deg,
            closure_rate_ms=-vel_behind,
            vel_cross_m_s=vel_cross,
            vel_below_m_s=vel_below,
            source=source,
        )

    def _compute_geometry(self) -> Optional[RelativeGeometry]:
        """Compute relative geometry from actual telemetry positions."""
        if self._follower_state is None or self._leader_state is None:
            return None
        try:
            # F1-alt 2026-05-09: bypass GeometryCalculator.compute_relative to
            # avoid env-side mutation of the controller's RelativePoseFusion
            # (P2_FUSION_CALL_ORDER_MUTATION_CONFIRMED in
            # analysis/p2_fusion_call_order_20260509T050603Z/P2_VERDICT.md).
            # Helper replicates the boom-tip projection math without driving
            # fusion.update_gps / fusion.update_uwb / fusion.get_state.
            # T-71 ENV ALIGN2 (flag-gated; OFF => operands byte-identical):
            # dead-reckon the leader wire sample to the FOLLOWER POSITION
            # epoch (self._follower_lat/lon come from
            # ctrl.follower._last_position, :4144 — so follower_pos_tbms is
            # the pairing epoch, NOT the attitude stamp raw uses). Reuses the
            # #337/#475 helpers unchanged; a bespoke re-derivation was
            # REFUTED offline (sd 1.23 -> 1.51, ENV_GEOM_RCA.md).
            _env_lead_lat, _env_lead_lon = self._leader_lat, self._leader_lon
            self._env_align_mode = 2          # 2 = unaligned passthrough
            self._env_align_age_s = 0.0
            if _FD_ENV_ALIGN2:
                try:
                    _elrs = self._ctrl_ref.leader_rx.last_state
                    _el_tbms = int(getattr(_elrs, "time_boot_ms", -1) or -1)
                    _efp = getattr(getattr(self._ctrl_ref, "follower", None),
                                   "_last_position", None)
                    _ef_tbms = int(getattr(_efp, "time_boot_ms", -1) or -1)
                    (_el_lat, _el_lon, _el_vn,
                     _el_ve) = _coherent_leader_operands(
                        _elrs, self._leader_lat, self._leader_lon,
                        self._leader_vn, self._leader_ve)
                    _e_age = _align_age_from_tbms(_ef_tbms, _el_tbms)
                    if _e_age > 0.0:          # leader stale => advance
                        _env_lead_lat, _env_lead_lon = _align_leader_sample(
                            _el_lat, _el_lon, _el_vn, _el_ve, _e_age)
                        self._env_align_mode = 0
                        self._env_align_age_s = _e_age
                    elif _e_age < 0.0:        # leader newer => rewind
                        _env_lead_lat, _env_lead_lon = _align_leader_sample(
                            _el_lat, _el_lon, -_el_vn, -_el_ve, -_e_age)
                        self._env_align_mode = 0
                        self._env_align_age_s = _e_age
                except Exception:             # fail closed to legacy pairing
                    _env_lead_lat, _env_lead_lon = (self._leader_lat,
                                                    self._leader_lon)
                    self._env_align_mode = 2
                    self._env_align_age_s = 0.0
            geom = self._project_to_boom_frame(
                foll_lat=self._follower_lat,
                foll_lon=self._follower_lon,
                foll_alt=self._follower_state.alt_msl_m,
                foll_vn=self._follower_vn,
                foll_ve=self._follower_ve,
                foll_vd=self._follower_state.vd_ms,
                lead_lat=_env_lead_lat,
                lead_lon=_env_lead_lon,
                lead_alt=self._leader_state.alt_msl_m,
                lead_vn=self._leader_vn,
                lead_ve=self._leader_ve,
                lead_vd=self._leader_state.vd_ms,
                lead_heading_deg=self._leader_state.heading_deg,
                boom_length_m=fc.BOOM_TIP_NOMINAL_BEHIND_M,
                source="ENV_GPS_DIRECT",
            )
            self._last_geom = geom

            # Patch C2 2026-05-06: RAW_TRUTH geometry from SIM_STATE follower
            # position + leader_rx leader position. Same boom-tip projection as
            # ENV_FUSED (BOOM_TIP_NOMINAL_BEHIND_M). Leader truth = leader_rx
            # (leader is structurally non-SITL — no leader SIM_STATE per
            # project_option_c_phase36c). RAW vs ENV delta isolates follower-
            # side GPS noise + EKF filter view; leader feed is identical.
            if (self._sim_state_lat is not None
                    and self._sim_state_lon is not None
                    and self._sim_state_alt_m is not None):
                try:
                    # F1-alt 2026-05-09: same fusion-bypass as ENV channel
                    # above. Critically, this prevents the previous
                    # compute_relative call from injecting SIM_STATE truth as
                    # a "GPS" measurement into the fusion (the dominant P2
                    # contamination source — see P2_VERDICT.md §3-§4).
                    # Source label switches "FUSED" -> "RAW_TRUTH_DIRECT".
                    # #337/#339: ALIGN2 (default-ON) dead-reckons the leader
                    # sample to the SIM_STATE's sender-side sample time.
                    # Kill-switch =0 => lead_lat/lon unchanged (the original
                    # receive-side pairing; the #334 v1 align was DELETED #342).
                    _lead_lat_a, _lead_lon_a = self._leader_lat, self._leader_lon
                    _hold_raw = False
                    _age2 = float("nan")       # #442: alignment age (snapshot)
                    _lead_tbms = -1            # #442: leader operand stamp (snapshot)
                    # #343: default provenance = unaligned passthrough; the
                    # ALIGN2 paths below overwrite with 0 (fresh) or 1 (held).
                    self._raw_align_mode = 2
                    if _FD_ENV_RAW_TRUTH_ALIGN2:
                        # #337 Contract A: epoch = the SIM_STATE's SENDER-SIDE
                        # sample time (adjacent-ATTITUDE tbms). Advance the
                        # leader to that instant (symmetric: backward when the
                        # leader sample is NEWER). Follower truth untouched.
                        # No stamp (att_tbms<=0) => HOLD the previous stamped
                        # sample (reuse the previous aligned raw_geom, counted;
                        # <=50 ms stale, consistent — per the Contract A spec).
                        # First-cut passthrough here re-contaminated 13-15 % of
                        # rows (B0v2/B1v2 measured) — spec-correct hold fixes it.
                        if (self._sim_state_att_tbms <= 0
                                and self._raw_truth_geom is not None):
                            self._raw_align2_held += 1
                            _hold_raw = True  # keep previous aligned raw_geom
                            self._raw_align_mode = 1  # #343: held reuse
                        try:
                            _lrs = self._ctrl_ref.leader_rx.last_state
                            _lead_tbms = getattr(_lrs, "time_boot_ms", -1)
                            # #475: operand + DR velocity from the SAME grab
                            # as the stamp (coherent single-object read);
                            # kill-switch reverts to the pre-fix mixed reads.
                            if _FD_ENV_ALIGN2_COHERENT:
                                (_al_lat, _al_lon, _al_vn,
                                 _al_ve) = _coherent_leader_operands(
                                    _lrs, self._leader_lat, self._leader_lon,
                                    self._leader_vn, self._leader_ve)
                            else:
                                _al_lat, _al_lon = (self._leader_lat,
                                                    self._leader_lon)
                                _al_vn, _al_ve = (self._leader_vn,
                                                  self._leader_ve)
                            # F-RAW (T-92): epoch-compensate the truth-feed
                            # stamp INSIDE the age only (flag-gated, default
                            # OFF; meta stamps below stay RAW).
                            if (_FD_RAW_EPOCH_COMP
                                    and self._sim_state_att_tbms > 0):
                                _att_eff = self._repoch_apply(
                                    self._sim_state_att_tbms, _lead_tbms)
                            else:
                                _att_eff = self._sim_state_att_tbms
                            _age2 = _align_age_from_tbms(
                                _att_eff, _lead_tbms)
                            # FIXA (flag-gated default OFF): center the
                            # lattice for the ALIGN DECISION ONLY; _age2
                            # itself stays RAW for meta/CSV/probes.
                            if _FD_RAW_ALIGN_CENTER:
                                _age2_eff = _center_age(
                                    _age2, self._align_center_win)
                            else:
                                _age2_eff = _age2
                            if _FD_EPOCH_PROBE:   # T-165; never raises
                                self._epoch_probe_emit(_lead_tbms, _age2)
                            if self._sim_state_att_tbms <= 0:
                                pass  # held above (or startup passthrough)
                            elif _age2 >= 0.0:
                                _lead_lat_a, _lead_lon_a = _align_leader_sample(
                                    _al_lat, _al_lon, _al_vn, _al_ve,
                                    _age2_eff)
                                self._raw_align2_used += 1
                                self._raw_align_mode = 0  # #343: fresh-aligned
                            else:
                                # leader NEWER than the SIM_STATE sample:
                                # dead-reckon BACKWARD by |age| (same math,
                                # negated velocity; v1 helper untouched).
                                _lead_lat_a, _lead_lon_a = _align_leader_sample(
                                    _al_lat, _al_lon, -_al_vn, -_al_ve, -_age2)
                                self._raw_align2_used += 1
                                self._raw_align_mode = 0  # #343: fresh-aligned
                        except Exception:
                            _lead_lat_a, _lead_lon_a = self._leader_lat, self._leader_lon
                            self._raw_align_mode = 2  # #343: fell back unaligned
                        # #475 R3 provenance: geometry this tick rides a
                        # HELD follower sample (consume seam rejecting).
                        if (_FD_ENV_ALIGN2_COHERENT
                                and self._r3_consec_rejects > 0
                                and self._raw_align_mode == 0):
                            self._raw_align_mode = 4
                        # #475 R4 provenance: fresh-aligned on a SYNTHESIZED
                        # sender stamp (kinematic tick-snap). Mode-4 hold
                        # keeps precedence while the seam is rejecting.
                        if (self._sim_state_att_synth
                                and self._raw_align_mode == 0):
                            self._raw_align_mode = 5
                        # FM-A1 disclosure (rate-limited by count multiples).
                        # T-225: the R3/R4/R5 counters ride the SAME emit so the
                        # stamp-provenance mix is attributable per leg. mode-5
                        # (synth-aligned) is reachable by TWO routes that the
                        # CSV cannot tell apart — R4 (pairer returned no stamp)
                        # vs R5 (a present stamp failed pairing-consistency and
                        # was re-stamped) — and separating them is the open
                        # question for T-93 leg 78013 (88.3% mode-5 on a CLEAN
                        # emission lag). Counters are pre-existing; only this
                        # line changes. No new state, no schema change.
                        _tot2 = self._raw_align2_used + self._raw_align2_held
                        if _tot2 and _tot2 % 2000 == 0:
                            logger.info(
                                "[RAW_ALIGN2] used=%d held=%d "
                                "(held-fraction %.1f%%) r3_rej=%d r4_acc=%d "
                                "r4_dec=%d r5_restamp=%d",
                                self._raw_align2_used, self._raw_align2_held,
                                100.0 * self._raw_align2_held / _tot2,
                                self._r3_pair_rejects, self._r4_synth_accepts,
                                self._r4_synth_declines, self._r5_restamps)
                    if not _hold_raw:
                        raw_geom = self._project_to_boom_frame(
                            foll_lat=self._sim_state_lat,
                            foll_lon=self._sim_state_lon,
                            foll_alt=self._sim_state_alt_m,
                            foll_vn=self._follower_vn,
                            foll_ve=self._follower_ve,
                            foll_vd=self._follower_state.vd_ms,
                            lead_lat=_lead_lat_a,
                            lead_lon=_lead_lon_a,
                            lead_alt=self._leader_state.alt_msl_m,
                            lead_vn=self._leader_vn,
                            lead_ve=self._leader_ve,
                            lead_vd=self._leader_state.vd_ms,
                            lead_heading_deg=self._leader_state.heading_deg,
                            boom_length_m=fc.BOOM_TIP_NOMINAL_BEHIND_M,
                            source="RAW_TRUTH_DIRECT",
                        )
                        # #475 R2: write-time plausibility hold.
                        # A behind step >0.8 m within one <=0.06 s tick is
                        # kinematically impossible => pairing-inconsistent row
                        # (epoch stutters surviving R1). Contract-A remedy:
                        # HOLD previous aligned geom, counted, provenance
                        # mode 3; RELEASED after 4 consecutive (a sustained
                        # real anomaly is accepted, never masked). Same
                        # _FD_ENV_ALIGN2_COHERENT kill-switch as R1.
                        _now_t = (self._time_auth.now_s()
                                  if self._time_auth else None)
                        if (_FD_ENV_ALIGN2_COHERENT
                                and self._raw_truth_geom is not None
                                and self._raw_plaus_consec < 4
                                and _raw_step_implausible(
                                    self._raw_plaus_prev_b,
                                    self._raw_plaus_prev_t,
                                    raw_geom.behind_m, _now_t)):
                            self._raw_plaus_held += 1
                            self._raw_plaus_consec += 1
                            self._raw_align_mode = 3  # plausibility-hold
                            # previous aligned raw_geom retained
                            # #475 R2b: advance the comparison clock through
                            # the hold (held value IS our estimate at now) so
                            # the NEXT check stays single-tick — without this,
                            # post-hold dt >0.06 blinded the gate and admitted
                            # the artifact (180/180 residual jumps = 1.236k
                            # quanta, ~85% immediately post-hold [MEASURED]).
                            if _now_t is not None:
                                self._raw_plaus_prev_t = _now_t
                        else:
                            self._raw_plaus_consec = 0
                            self._raw_truth_geom = raw_geom
                            if _now_t is not None:
                                self._raw_plaus_prev_b = raw_geom.behind_m
                                self._raw_plaus_prev_t = _now_t
                    # _hold_raw: previous aligned raw_geom retained (#337).
                    # #442: compute-TIME operand-provenance snapshot (NOT re-read
                    # at write time — the writer re-reads stamps, which mis-attributes
                    # mode-1 held rows). Writer-only; never enters any training surface.
                    # Aligned leader pos (rm_lead_*_al) = exactness-witness input;
                    # raw leader pos+vel+stamp = offline de-ZOH re-interpolation input.
                    _gc = self._geom_calc
                    self._raw_truth_meta = {
                        "foll_tbms": self._sim_state_att_tbms,
                        "foll_wire": self._sim_state_wire_stamp,
                        "foll_lat": self._sim_state_lat,
                        "foll_lon": self._sim_state_lon,
                        "foll_alt": self._sim_state_alt_m,
                        "foll_vn": self._sim_state_vn,
                        "foll_ve": self._sim_state_ve,
                        "foll_vd": self._sim_state_vd,
                        "lead_tbms": _lead_tbms,
                        "lead_lat": self._leader_lat,
                        "lead_lon": self._leader_lon,
                        "lead_alt": self._leader_state.alt_msl_m,
                        "lead_vn": self._leader_vn,
                        "lead_ve": self._leader_ve,
                        "lead_hdg": self._leader_state.heading_deg,
                        "lead_lat_al": _lead_lat_a,
                        "lead_lon_al": _lead_lon_a,
                        "age2_s": _age2,
                        "home_lat": getattr(_gc, "home_lat", float("nan")),
                        # F-RAW provenance (flag OFF => empty columns; every
                        # legacy column stays byte-identical, #442 convention).
                        "repoch_state": (self._repoch_state
                                         if _FD_RAW_EPOCH_COMP else None),
                        "repoch_ms": (self._repoch_ms
                                      if _FD_RAW_EPOCH_COMP else None),
                    }
                except Exception:
                    # RAW_TRUTH derivation is observability-only.
                    self._raw_truth_geom = None
                    self._raw_align_mode = -1  # #343: no raw geom
                    self._raw_truth_meta = None
            else:
                self._raw_truth_geom = None
                self._raw_align_mode = -1  # #343: no raw geom
            if self._lead_foll_diag_enabled:
                try:
                    lead_hdg = self._leader_state.heading_deg
                    foll_hdg = self._follower_state.heading_deg
                    hdg_delta = ((lead_hdg - foll_hdg + 540.0) % 360.0) - 180.0
                    rel_vn = self._follower_vn - self._leader_vn
                    rel_ve = self._follower_ve - self._leader_ve
                    rel_vd = (
                        self._follower_state.vd_ms - self._leader_state.vd_ms
                    )
                    try:
                        fsm = self._ctrl_ref.get_fsm_state().name
                    except Exception:
                        fsm = "NONE"
                    fusion = getattr(self._geom_calc, "fusion", None)
                    gps_resnorm = (
                        fusion.last_gps_position_residual_norm_m
                        if (fusion is not None
                            and fusion.last_gps_position_residual_norm_m is not None)
                        else float("nan")
                    )
                    uwb_innov = (
                        fusion.last_uwb_innovation_m
                        if (fusion is not None
                            and fusion.last_uwb_innovation_m is not None)
                        else float("nan")
                    )
                    uwb_reason = (
                        fusion.last_uwb_accept_reject_reason
                        if (fusion is not None
                            and fusion.last_uwb_accept_reject_reason is not None)
                        else "NONE"
                    )
                    cos_hdg = math.cos(math.radians(lead_hdg))
                    sin_hdg = math.sin(math.radians(lead_hdg))
                    vb_recomputed = -(rel_vn * cos_hdg + rel_ve * sin_hdg)
                    vc_recomputed = -rel_vn * sin_hdg + rel_ve * cos_hdg
                    vd_recomputed = rel_vd
                    logger.info(
                        "[LEAD_FOLL_DIAG] env_id=%d t_sim=%.3f fsm=%s "
                        "lead_hdg=%.2f foll_hdg=%.2f hdg_delta=%.2f "
                        "lead_vn=%.3f lead_ve=%.3f lead_vd=%.3f "
                        "foll_vn=%.3f foll_ve=%.3f foll_vd=%.3f "
                        "rel_vn=%.3f rel_ve=%.3f rel_vd=%.3f "
                        "vel_behind_geom=%.3f vel_cross_geom=%.3f vel_below_geom=%.3f "
                        "vel_behind_recomputed=%.3f vel_cross_recomputed=%.3f vel_below_recomputed=%.3f "
                        "behind=%.3f cross=%.3f below=%.3f range=%.3f "
                        "src=%s gps_resnorm=%.3f uwb_innov=%.3f uwb_reason=%s",
                        self.env_id, self._time_auth.now_s(), fsm,
                        lead_hdg, foll_hdg, hdg_delta,
                        self._leader_vn, self._leader_ve,
                        self._leader_state.vd_ms,
                        self._follower_vn, self._follower_ve,
                        self._follower_state.vd_ms,
                        rel_vn, rel_ve, rel_vd,
                        -geom.closure_rate_ms,
                        geom.vel_cross_m_s,
                        geom.vel_below_m_s,
                        vb_recomputed, vc_recomputed, vd_recomputed,
                        geom.behind_m, geom.cross_m, geom.below_m,
                        geom.range_m,
                        geom.source, gps_resnorm, uwb_innov, uwb_reason,
                    )
                except Exception:
                    logger.debug("LEAD_FOLL_DIAG emission failed", exc_info=True)
                # Tier-2C [FUSION_DIAG] emission — covariance snapshots and
                # state-before/after for SMOKE-A2-B3 covariance discriminator.
                # Pure additive log; no behavior change. Gated by the same
                # FLIGHTDOCK_LEADFOLL_DIAG=1 flag.
                try:
                    def _csv6(v):
                        if v is None:
                            return "nan,nan,nan,nan,nan,nan"
                        return ",".join(f"{x:.6g}" for x in v)

                    if fusion is not None:
                        P = fusion.P
                        cur_pdiag = [P[i][i] for i in range(6)]
                        gps_pre_p = fusion.last_gps_P_pre_update_diag
                        uwb_pre_p = fusion.last_uwb_P_pre_gate_diag
                        uwb_post_p = fusion.last_uwb_P_post_update_diag
                        gps_xb = fusion.last_gps_state_before_update
                        gps_xa = fusion.last_gps_state_after_update
                        uwb_xb = fusion.last_uwb_filter_state_before_update
                        uwb_xa = fusion.last_uwb_filter_state_after_update
                        gps_velresnorm = (
                            fusion.last_gps_velocity_residual_norm_mps
                            if fusion.last_gps_velocity_residual_norm_mps is not None
                            else float("nan")
                        )
                        uwb_gate_thr = (
                            fusion.last_uwb_gate_threshold_m
                            if fusion.last_uwb_gate_threshold_m is not None
                            else float("nan")
                        )
                        uwb_htph = (
                            fusion.last_uwb_HtPH_pre_gate
                            if fusion.last_uwb_HtPH_pre_gate is not None
                            else float("nan")
                        )
                        gps_ts = (
                            fusion.last_gps_measurement_timestamp_s
                            if fusion.last_gps_measurement_timestamp_s is not None
                            else float("nan")
                        )
                        uwb_ts = (
                            fusion.last_uwb_measurement_timestamp_s
                            if fusion.last_uwb_measurement_timestamp_s is not None
                            else float("nan")
                        )
                        # Class B remediation, Commit B1: monotonic UWB
                        # event counters (accept/reject), so the SMOKE-A2-B4
                        # parser can count UWB events from per-tick counter
                        # edges instead of latched-snapshot scans
                        # (which over-count by env/UWB rate ratio per the
                        # SMOKE-A2-B3 "14 vs 0" pattern).
                        fusion_uwb_acc = fusion.uwb_update_count
                        fusion_uwb_rej = fusion.uwb_rejected_count
                    else:
                        cur_pdiag = None
                        gps_pre_p = uwb_pre_p = uwb_post_p = None
                        gps_xb = gps_xa = uwb_xb = uwb_xa = None
                        gps_velresnorm = uwb_gate_thr = uwb_htph = float("nan")
                        gps_ts = uwb_ts = float("nan")
                        fusion_uwb_acc = -1
                        fusion_uwb_rej = -1

                    logger.info(
                        "[FUSION_DIAG] env_id=%d t_sim=%.3f "
                        "P_diag_now=%s "
                        "P_pre_gps=%s P_pre_uwb_gate=%s P_post_uwb=%s "
                        "HtPH_pre_uwb=%.6g "
                        "gps_state_before=%s gps_state_after=%s "
                        "uwb_state_before=%s uwb_state_after=%s "
                        "gps_resnorm=%.6g gps_velresnorm=%.6g "
                        "uwb_innov=%.6g uwb_gate_thr=%.6g uwb_reason=%s "
                        "last_gps_ts=%.3f last_uwb_ts=%.3f "
                        "fusion_uwb_update_count=%d fusion_uwb_rejected_count=%d",
                        self.env_id, self._time_auth.now_s(),
                        _csv6(cur_pdiag),
                        _csv6(gps_pre_p), _csv6(uwb_pre_p), _csv6(uwb_post_p),
                        uwb_htph,
                        _csv6(gps_xb), _csv6(gps_xa),
                        _csv6(uwb_xb), _csv6(uwb_xa),
                        gps_resnorm, gps_velresnorm,
                        uwb_innov, uwb_gate_thr, uwb_reason,
                        gps_ts, uwb_ts,
                        fusion_uwb_acc, fusion_uwb_rej,
                    )
                except Exception:
                    logger.debug("FUSION_DIAG emission failed", exc_info=True)
            return geom
        except Exception:
            logger.debug("Failed to compute geometry, using last known", exc_info=True)
            return self._last_geom

    # ------------------------------------------------------------------
    # Patch C2 2026-05-06: Paired-channel CSV (RAW_TRUTH / ENV_FUSED / CTRL_FUSED)
    # ------------------------------------------------------------------

    _PAIRED_CSV_FIELDS = [
        "timestamp_wall_s", "sim_time_s", "episode_idx", "env_id", "tick_idx",
        "fsm_state",
        "resolved_wind_max_ms", "resolved_wind_turb", "resolved_gps_accuracy",
        "resolved_leader_profile", "harness_uwb_enable", "reward_source_mode",
        "condition_profile",
        "raw_behind_m", "raw_cross_m", "raw_below_m", "raw_range_m",
        "env_behind_m", "env_cross_m", "env_below_m", "env_range_m",
        "ctrl_behind_m", "ctrl_cross_m", "ctrl_below_m", "ctrl_range_m",
        "ctrl_source",
        "delta_env_raw_behind_m", "delta_env_raw_cross_m",
        "delta_env_raw_below_m", "delta_env_raw_range_m",
        "delta_ctrl_raw_behind_m", "delta_ctrl_raw_cross_m",
        "delta_ctrl_raw_below_m", "delta_ctrl_raw_range_m",
        "raw_m3b", "env_m3b", "ctrl_m3b",
        # #447: legacy FormationSuccessMetrics m3b_in_envelope reading — divergence
        # witness for the starved observer feed (Ledger #11). ctrl_m3b (above) is now
        # a DIRECT live-geometry box check and is the certification channel; this
        # column is forensics only. Grouped with its sibling; rm_* stays the last block.
        "ctrl_m3b_sm",
        "uwb_update_count", "uwb_rejected_count", "uwb_last_innovation_m",
        "uwb_last_t_s",
        # C2.1a passive instrumentation (M3 hypothesis).
        "leader_msg_time_boot_ms", "follower_time_boot_ms",
        "leader_age_ms", "leader_freshness_passed",
        "follower_pos_time_boot_ms", "follower_pos_age_ms",
        # Option-A1 cache-age telemetry (2026-05-09). Triage gate excludes
        # rows where ctrl_cache_is_stale=1 from RAW-CTRL pass/fail since the
        # ctrl_*_m columns read from _ctrl_ref._geom_cache between controller
        # ticks. Fresh-cache rows (is_stale=0) are authoritative.
        "ctrl_cache_age_rows", "ctrl_cache_age_s", "ctrl_cache_is_stale",
        # F1-alt source labels (2026-05-09). Post-patch:
        #   env_geom_source = "ENV_GPS_DIRECT" (direct GPS-position projection,
        #     NOT fusion-after-update)
        #   raw_geom_source = "RAW_TRUTH_DIRECT" (SIM_STATE projection, NOT
        #     fusion-after-truth-injection)
        # If either column shows "FUSED" the F1-alt patch is not active.
        "env_geom_source", "raw_geom_source",
        # #343 v2.1: provenance of the raw_* fields this row — 0=fresh
        # sender-aligned, 1=held reuse (#337 Contract A hold), 2=unaligned
        # passthrough, -1=no raw geom. APPENDED LAST; held rows PRESERVED
        # (mark-don't-drop, same pattern as ctrl_cache_is_stale).
        # T-71 ENV ALIGN2 provenance, grouped with raw_align_mode (both are
        # alignment provenance) and placed so that BOTH pinned invariants
        # survive untouched: the rm_* block stays LAST, and raw_align_mode
        # remains the last non-rm column immediately preceding it (three
        # independent tests assert these; the suite caught an earlier append
        # that violated them — honoured, not weakened). Values-only; every
        # legacy column byte-identical.
        # env_align_mode: 0=aligned this tick, 2=unaligned passthrough (flag OFF
        # or fail-closed). env_align_age_s: applied pairing age (signed; negative
        # = leader newer, rewound). Independent witness that alignment ACTUALLY
        # fired — never infer activity from the delta it is meant to shrink
        # (#465b readback-vs-witness lesson).
        "env_align_mode", "env_align_age_s",
        "raw_align_mode",
        # #442: raw-truth OPERAND provenance snapshot (compute-time, writer-only;
        # offline de-ZOH instrument input, #443/#444). Additive/APPENDED-LAST; every
        # legacy column above is byte-identical. NEVER enters any training surface.
        # foll_* = SIM_STATE follower operand (TRUE wire velocities); lead_* = raw
        # leader operand; lead_*_al = ALIGNED leader pos actually projected (exactness
        # witness); age2_s = ALIGN2 age; home_lat = projection datum (east cos scale).
        "rm_foll_tbms", "rm_foll_wire", "rm_foll_lat", "rm_foll_lon", "rm_foll_alt",
        "rm_foll_vn", "rm_foll_ve", "rm_foll_vd",
        "rm_lead_tbms", "rm_lead_lat", "rm_lead_lon", "rm_lead_alt",
        "rm_lead_vn", "rm_lead_ve", "rm_lead_hdg",
        "rm_lead_lat_al", "rm_lead_lon_al", "rm_age2_s", "rm_home_lat",
        # F-RAW (T-92): epoch-compensation provenance — APPENDED LAST inside
        # the rm_* block (rm_* stays the final block; raw_align_mode still
        # immediately precedes it; schema-additive per the #442 convention).
        # state: 0 measuring, 1 latched-none, 2 latched-comp, -1 fail-closed;
        # empty when FD_RAW_EPOCH_COMP is OFF.
        "rm_repoch_state", "rm_repoch_ms",
    ]
    # T-392a: the epoch-stamp witness is written to its OWN file, NOT appended
    # here. Three tests pin "the rm_* block stays LAST" and "raw_align_mode is
    # the last non-rm column"; the first draft of this dispatch appended after
    # rm_* and the suite caught it. Honouring those invariants rather than
    # weakening them keeps the paired CSV BYTE-IDENTICAL (a real zero-delta
    # claim), and the side file joins on (episode_idx, tick_idx, sim_time_s).
    _ESTAMP_CSV_FIELDS = [
        "sim_time_s", "episode_idx", "tick_idx",
        "es_lead_wire",    # leader wire tbms carried by this row's sample
        "es_lead_latch",   # RAW ATTITUDE tbms the broadcaster actually latched
        "es_foll_latch",   # the env's own adjacent-ATTITUDE latch
    ]

    def _epoch_probe_emit(self, lead_tbms, age2):
        """T-165 epoch probe seam line (flag-gated by the caller; default
        OFF). First _EPOCH_PROBE_N emits per episode; counts EVERY emit so a
        boot full of sentinels still terminates the budget. Observability
        only: reads instance state, writes nothing back, NEVER raises (E4 —
        a probe that can kill a leg is worse than no probe)."""
        try:
            n = getattr(self, "_epoch_probe_count", 0)
            if n >= _EPOCH_PROBE_N:
                return
            self._epoch_probe_count = n + 1
            foll = getattr(self, "_sim_state_att_tbms", -1)
            d_ms = (foll - lead_tbms) if (foll > 0 and lead_tbms > 0) else "NA"
            logger.info(
                "[EPOCH_PROBE] seam foll_tbms=%s lead_tbms=%s d_ms=%s "
                "age2_s=%s wire=%s mode=%s",
                foll, lead_tbms, d_ms, age2,
                getattr(self, "_sim_state_wire_stamp", None),
                getattr(self, "_raw_align_mode", None))
        except Exception:
            pass

    def _repoch_apply(self, att_tbms, lead_tbms):
        """F-RAW v2 (T-92 + frv AMBER, flag FD_RAW_EPOCH_COMP): compensate
        the boot-latched truth-feed epoch offset inside the ALIGN2 age ONLY.
        DUAL-WINDOW latch: window A = valid pairs 1-32, window B = pairs
        201-232 (~12 s apart). Latch comp only when both windows agree on
        the same stable snapped offset; snapped 0/0 latches none; any
        disagreement or instability FAILS CLOSED (state -1, marked, never
        compensated) — the measured transient class lands there. After a
        comp latch, a rolling guard re-estimates every _REPOCH_N pairs and
        demotes to -1 on 2 consecutive off-latch windows (transients longer
        than window B). Meta/CSV stamps stay RAW so the offline T-93
        classifier still measures the wire truth. Observability-only."""
        if att_tbms <= 0 or lead_tbms <= 0:
            return (att_tbms - self._repoch_ms
                    if self._repoch_state == 2 else att_tbms)
        if self._repoch_state == 2:
            # rolling consistency guard over trailing _REPOCH_N raw deltas
            self._repoch_deltas.append(float(att_tbms - lead_tbms))
            if len(self._repoch_deltas) >= _REPOCH_N:
                med, mad, snapped, _ = _repoch_estimate(self._repoch_deltas)
                self._repoch_deltas = []
                if float(snapped) != self._repoch_ms:
                    self._repoch_guard_trips += 1
                    if self._repoch_guard_trips >= 2:
                        logger.info(
                            "[RAW_EPOCH] GUARD DEMOTE: latched=%.0f window "
                            "median=%.1f snapped=%d — state -1, comp OFF",
                            self._repoch_ms, med, snapped)
                        self._repoch_state = -1
                        self._repoch_ms = 0.0
                        return att_tbms
                else:
                    self._repoch_guard_trips = 0
            return att_tbms - self._repoch_ms
        if self._repoch_state != 0:
            return att_tbms  # latched-none or fail-closed: RAW
        self._repoch_pairs += 1
        p = self._repoch_pairs
        if p <= _REPOCH_N:                       # window A
            self._repoch_deltas.append(float(att_tbms - lead_tbms))
            if p == _REPOCH_N:
                self._repoch_win_a = _repoch_estimate(self._repoch_deltas)
                self._repoch_deltas = []
        elif p > _REPOCH_B_START:                # window B
            self._repoch_deltas.append(float(att_tbms - lead_tbms))
            if len(self._repoch_deltas) >= _REPOCH_N:
                b = _repoch_estimate(self._repoch_deltas)
                self._repoch_deltas = []
                a = self._repoch_win_a
                agree = (a[3] == b[3] and a[2] == b[2])
                if agree and a[3] == "none":
                    self._repoch_state = 1
                elif agree and a[3] == "comp":
                    self._repoch_state = 2
                    self._repoch_ms = float(a[2])
                else:
                    self._repoch_state = -1  # transient/unstable/late-onset
                logger.info(
                    "[RAW_EPOCH] A(med=%.1f mad=%.1f snap=%d %s) "
                    "B(med=%.1f mad=%.1f snap=%d %s) -> state=%d comp=%.0f",
                    a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3],
                    self._repoch_state, self._repoch_ms)
                if self._repoch_state == 2:
                    return att_tbms - self._repoch_ms
        return att_tbms

    def _open_paired_csv(self) -> None:
        """Open per-episode paired_channels.csv. Output dir from env or default."""
        try:
            out_root = os.environ.get("FLIGHTDOCK_PAIRED_CSV_DIR")
            if not out_root:
                out_root = os.environ.get("FLIGHTDOCK_LOG_DIR", "logs")
            out_dir = Path(out_root) / f"env{self.env_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"paired_channels_ep{self._paired_episode_idx:04d}.csv"
            # Close previous before opening new.
            self._close_paired_csv()
            f = open(path, "w", newline="")
            w = csv.DictWriter(f, fieldnames=self._PAIRED_CSV_FIELDS)
            w.writeheader()
            self._paired_csv_file = f
            self._paired_csv_writer = w
            self._paired_csv_path = path
        except Exception:
            logger.debug("Failed to open paired_channels.csv", exc_info=True)
            self._paired_csv_file = None
            self._paired_csv_writer = None
            self._paired_csv_path = None

    def _close_paired_csv(self) -> None:
        # T-392a: the witness file is opened lazily beside the paired CSV, so
        # it closes with it (one lifetime, no leak across episodes).
        _ef = getattr(self, "_estamp_csv_file", None)
        if _ef is not None:
            try:
                _ef.flush()
                _ef.close()
            except Exception:
                pass
        self._estamp_csv_file = None
        self._estamp_csv_writer = None
        if self._paired_csv_file is not None:
            try:
                self._paired_csv_file.flush()
                self._paired_csv_file.close()
            except Exception:
                logger.debug("Failed to close paired_channels.csv", exc_info=True)
        self._paired_csv_file = None
        self._paired_csv_writer = None

    def _fs_res_dict(self, applied):
        """Residual columns: policy AND applied, per RESLOG-749's mandate that
        residual actions be reviewable. Zero-residual legs make these equal;
        ARM-F will not."""
        out = {}
        pol = getattr(self, "_fs_policy_action", None)
        for i in range(3):
            if applied is not None:
                out["res_applied_%d" % i] = float(applied[i])
            if pol is not None:
                out["res_policy_%d" % i] = float(pol[i])
        out["res_source"] = getattr(self, "_fs_res_source", "")
        return out

    def _open_flight_state(self) -> None:
        """TELEM-766-A: open this episode's flight-state record.

        Sibling of the paired CSV, never an extension of it: that schema is
        frozen and its readers must not be broken."""
        try:
            root = (os.environ.get("FLIGHTDOCK_PAIRED_CSV_DIR")
                    or os.environ.get("FLIGHTDOCK_LOG_DIR", "logs"))
            self._fs_rec = flight_state_recorder.open_leg(
                Path(root) / f"env{self.env_id}", self._paired_episode_idx)
        except Exception:
            logger.debug("flight_state open failed", exc_info=True)
            self._fs_rec = None

    def _close_flight_state(self) -> None:
        """Flush and close. The flush is what makes a gzip file readable."""
        rec = getattr(self, "_fs_rec", None)
        if rec is not None:
            rec.close_leg()
        self._fs_rec = None

    def _write_flight_state_row(self, env_geom=None,
                                terminal_reason="") -> None:
        """Emit one flight-state tick. Copies only; derivation is offline."""
        rec = getattr(self, "_fs_rec", None)
        if rec is None:
            return
        try:
            ctrl = self._ctrl_ref
            f_mav = getattr(getattr(ctrl, "follower", None), "mav", None) or self._mav
            lrx = getattr(ctrl, "leader_rx", None) or getattr(self, "_leader_rx", None)
            try:
                fsm = ctrl.get_fsm_state().name if ctrl else ""
            except Exception:
                fsm = ""
            cmd, res = self._last_baseline_cmd, self._prev_applied_residual
            rec.write_row(
                ident={"timestamp_wall_s": time.time(),
                       "sim_time_s": (self._time_auth.now_s()
                                      if self._time_auth else None),
                       "episode_idx": self._paired_episode_idx,
                       "env_id": self.env_id, "tick_idx": self._tick_count,
                       "fsm_state": fsm, "home_alt_msl": self._home_alt_msl,
                       "terminal_reason": terminal_reason},
                f_mav=f_mav, l_mav=getattr(lrx, "mav", None),
                f_used=self._follower_state, l_used=self._leader_state,
                geoms={"raw": self._raw_truth_geom, "env": env_geom,
                       "ctrl": getattr(ctrl, "_geom_cache", None)},
                cmds={"cmd_speed_mps": getattr(cmd, "speed_ms", None),
                      "cmd_heading_deg": getattr(cmd, "heading_deg", None),
                      "cmd_alt_msl_m": getattr(cmd, "altitude_m", None)},
                res=self._fs_res_dict(res),
                ctrl=ctrl)
        except Exception:
            logger.debug("flight_state row failed", exc_info=True)

    def _write_paired_row(self, env_geom: Optional[RelativeGeometry],
                          in_m3b_env: bool) -> None:
        """Emit one paired-channel row. Called per sub-tick from step()."""
        if self._paired_csv_writer is None:
            return
        try:
            raw = self._raw_truth_geom
            ctrl_geom = None
            ctrl_source = ""
            ctrl_in_m3b = False
            ctrl_sm_in_m3b = False
            # C2.1a passive instrumentation (M3 hypothesis test) -- read-only;
            # no get_geometry() / cache mutation. Sentinel -1 on any failure.
            leader_msg_tbms = follower_tbms = follower_pos_tbms = -1
            leader_age_ms = leader_freshness_passed = follower_pos_age_ms = -1
            ctrl = self._ctrl_ref
            if ctrl is not None:
                try:
                    leader_msg_tbms = int(ctrl.leader_rx.last_state.time_boot_ms)
                except Exception:
                    pass
                try:
                    follower_pos_tbms = int(ctrl.follower._last_position.time_boot_ms)
                except Exception:
                    pass
            try:
                auth = self._time_auth
                if auth is not None and auth.is_ready():
                    follower_tbms = int(getattr(auth, "_follower_tbms", -1))
                    if leader_msg_tbms >= 0:
                        is_fresh, _ = auth.check_leader_freshness(
                            leader_msg_tbms, stale_thresh_ms=2000)
                        leader_freshness_passed = int(bool(is_fresh))
            except Exception:
                pass
            if leader_msg_tbms >= 0 and follower_tbms >= 0:
                leader_age_ms = follower_tbms - leader_msg_tbms
            if follower_pos_tbms >= 0 and follower_tbms >= 0:
                follower_pos_age_ms = follower_tbms - follower_pos_tbms
            if self._ctrl_ref is not None:
                ctrl_geom = getattr(self._ctrl_ref, "_geom_cache", None)
                if ctrl_geom is not None:
                    ctrl_source = str(getattr(ctrl_geom, "source", ""))
                # #447: the FormationSuccessMetrics observer feed is starved
                # (range tracker frozen at first tick -> m3b_in_envelope
                # identically False on docking legs); its reading is kept ONLY
                # as the ctrl_m3b_sm divergence witness, never certification.
                sm = getattr(self._ctrl_ref, "_success_metrics", None)
                if sm is not None:
                    try:
                        m3_state = sm.get_m3_per_tick_state()
                        ctrl_sm_in_m3b = bool(m3_state.get("m3b_in_envelope", False))
                    except Exception:
                        ctrl_sm_in_m3b = False

            # #447 CTRL M3b: direct box check on the LIVE ctrl geometry —
            # identical canonical corridor to raw_m3b below (C2.1a-corrected
            # 0.7/0.5/0.7). Certification channel independent of the observer.
            if ctrl_geom is not None:
                ctrl_in_m3b = (
                    abs(ctrl_geom.behind_m) <= 0.7
                    and abs(ctrl_geom.cross_m) <= 0.5
                    and abs(ctrl_geom.below_m) <= 0.7
                )

            # RAW_TRUTH M3b: |behind| <= 0.7 AND |cross| <= 0.5 AND |below| <= 0.7
            # (boom-tip frame; aligned with canonical M3b STRICT corridor — C2.1a
            # corrected below-axis defect 0.5 -> 0.7).
            raw_in_m3b = False
            if raw is not None:
                raw_in_m3b = (
                    abs(raw.behind_m) <= 0.7
                    and abs(raw.cross_m) <= 0.5
                    and abs(raw.below_m) <= 0.7
                )

            d_env_raw_b = (env_geom.behind_m - raw.behind_m) if (env_geom and raw) else None
            d_env_raw_c = (env_geom.cross_m - raw.cross_m) if (env_geom and raw) else None
            d_env_raw_d = (env_geom.below_m - raw.below_m) if (env_geom and raw) else None
            d_env_raw_r = (env_geom.range_m - raw.range_m) if (env_geom and raw) else None
            d_ctrl_raw_b = (ctrl_geom.behind_m - raw.behind_m) if (ctrl_geom and raw) else None
            d_ctrl_raw_c = (ctrl_geom.cross_m - raw.cross_m) if (ctrl_geom and raw) else None
            d_ctrl_raw_d = (ctrl_geom.below_m - raw.below_m) if (ctrl_geom and raw) else None
            d_ctrl_raw_r = (ctrl_geom.range_m - raw.range_m) if (ctrl_geom and raw) else None

            # Per-episode rolling deltas for callback aggregates.
            if d_env_raw_b is not None:
                self._paired_delta_env_raw_behind.append(d_env_raw_b)
            if d_env_raw_c is not None:
                self._paired_delta_env_raw_cross.append(d_env_raw_c)
            if d_env_raw_d is not None:
                self._paired_delta_env_raw_below.append(d_env_raw_d)
            if d_ctrl_raw_b is not None:
                self._paired_delta_ctrl_raw_behind.append(d_ctrl_raw_b)
            if d_ctrl_raw_c is not None:
                self._paired_delta_ctrl_raw_cross.append(d_ctrl_raw_c)
            if d_ctrl_raw_d is not None:
                self._paired_delta_ctrl_raw_below.append(d_ctrl_raw_d)
            # Hallucinated M3b = perceived in M3b but truth says no.
            if raw is not None:
                if in_m3b_env and not raw_in_m3b:
                    self._paired_hallucinated_env_m3b_count += 1
                if ctrl_in_m3b and not raw_in_m3b:
                    self._paired_hallucinated_ctrl_m3b_count += 1

            # Option-A1 cache-age telemetry (2026-05-09). The ctrl_*_m columns
            # read from self._ctrl_ref._geom_cache, which only refreshes when
            # the controller's tick_once fires. The env writes paired rows much
            # faster than the controller ticks, so most rows read a stale cache.
            # Mark stale rows so the triage gate can exclude them from RAW-CTRL
            # pass/fail. Stale rows are PRESERVED in the CSV.
            self._paired_total_row_count += 1
            ctrl_cache_age_rows = 0
            ctrl_cache_age_s = 0.0
            ctrl_cache_is_stale = 0
            cur_wall_t = time.time()
            cur_ctrl_b = ctrl_geom.behind_m if ctrl_geom else None
            if cur_ctrl_b is not None:
                if (self._ctrl_cache_last_behind_m is None
                        or abs(cur_ctrl_b - self._ctrl_cache_last_behind_m) > 1e-9):
                    self._ctrl_cache_last_behind_m = cur_ctrl_b
                    self._ctrl_cache_last_row_idx = self._paired_total_row_count
                    self._ctrl_cache_last_wall_t = cur_wall_t
                    # cache fresh this row
                else:
                    ctrl_cache_age_rows = (
                        self._paired_total_row_count
                        - self._ctrl_cache_last_row_idx
                    )
                    if self._ctrl_cache_last_wall_t is not None:
                        ctrl_cache_age_s = cur_wall_t - self._ctrl_cache_last_wall_t
                    ctrl_cache_is_stale = 1 if ctrl_cache_age_rows > 0 else 0

            # UWB diagnostics (controller-side fusion).
            uwb_uc = uwb_rc = -1
            uwb_innov = 0.0
            uwb_t = -1.0
            try:
                if self._ctrl_ref is not None:
                    fusion = getattr(getattr(self._ctrl_ref, "geom_calc", None),
                                     "fusion", None)
                    if fusion is not None:
                        uwb_uc = int(getattr(fusion, "uwb_update_count", 0))
                        uwb_rc = int(getattr(fusion, "uwb_rejected_count", 0))
                        uwb_innov = float(getattr(fusion, "last_innovation_range", 0.0))
                        uwb_t = float(getattr(fusion, "last_uwb_time", -1.0) or -1.0)
            except Exception:
                pass

            try:
                fsm = self._ctrl_ref.get_fsm_state().name if self._ctrl_ref else ""
            except Exception:
                fsm = ""

            row = {
                "timestamp_wall_s": time.time(),
                "sim_time_s": self._time_auth.now_s() if self._time_auth else None,
                "episode_idx": self._paired_episode_idx,
                "env_id": self.env_id,
                "tick_idx": self._tick_count,
                "fsm_state": fsm,
                "resolved_wind_max_ms": self._current_wind_max_ms,
                "resolved_wind_turb": self._current_wind_turb,
                "resolved_gps_accuracy": self._current_gps_accuracy,
                "resolved_leader_profile": self._current_leader_profile,
                "harness_uwb_enable": self._current_harness_uwb_enable,
                "reward_source_mode": self._current_reward_source_mode,
                "condition_profile": self._current_condition_profile or "",
                "raw_behind_m": raw.behind_m if raw else None,
                "raw_cross_m": raw.cross_m if raw else None,
                "raw_below_m": raw.below_m if raw else None,
                "raw_range_m": raw.range_m if raw else None,
                "env_behind_m": env_geom.behind_m if env_geom else None,
                "env_cross_m": env_geom.cross_m if env_geom else None,
                "env_below_m": env_geom.below_m if env_geom else None,
                "env_range_m": env_geom.range_m if env_geom else None,
                "ctrl_behind_m": ctrl_geom.behind_m if ctrl_geom else None,
                "ctrl_cross_m": ctrl_geom.cross_m if ctrl_geom else None,
                "ctrl_below_m": ctrl_geom.below_m if ctrl_geom else None,
                "ctrl_range_m": ctrl_geom.range_m if ctrl_geom else None,
                "ctrl_source": ctrl_source,
                "ctrl_cache_age_rows": ctrl_cache_age_rows,
                "ctrl_cache_age_s": ctrl_cache_age_s,
                "ctrl_cache_is_stale": ctrl_cache_is_stale,
                "env_geom_source": (env_geom.source if env_geom else ""),
                "raw_geom_source": (raw.source if raw else ""),
                "delta_env_raw_behind_m": d_env_raw_b,
                "delta_env_raw_cross_m": d_env_raw_c,
                "delta_env_raw_below_m": d_env_raw_d,
                "delta_env_raw_range_m": d_env_raw_r,
                "delta_ctrl_raw_behind_m": d_ctrl_raw_b,
                "delta_ctrl_raw_cross_m": d_ctrl_raw_c,
                "delta_ctrl_raw_below_m": d_ctrl_raw_d,
                "delta_ctrl_raw_range_m": d_ctrl_raw_r,
                "raw_m3b": int(raw_in_m3b),
                "env_m3b": int(in_m3b_env),
                "ctrl_m3b": int(ctrl_in_m3b),
                "ctrl_m3b_sm": int(ctrl_sm_in_m3b),
                "uwb_update_count": uwb_uc,
                "uwb_rejected_count": uwb_rc,
                "uwb_last_innovation_m": uwb_innov,
                "uwb_last_t_s": uwb_t,
                "leader_msg_time_boot_ms": leader_msg_tbms,
                "follower_time_boot_ms": follower_tbms,
                "leader_age_ms": leader_age_ms,
                "leader_freshness_passed": leader_freshness_passed,
                "follower_pos_time_boot_ms": follower_pos_tbms,
                "follower_pos_age_ms": follower_pos_age_ms,
                # #343: structural invariant — raw None always reads -1.
                "raw_align_mode": (
                    self._raw_align_mode if raw is not None else -1),
                # T-71: env-channel alignment provenance for THIS row.
                "env_align_mode": getattr(self, "_env_align_mode", 2),
                "env_align_age_s": round(
                    float(getattr(self, "_env_align_age_s", 0.0)), 6),
            }
            # #442: raw-operand provenance columns from the compute-time snapshot
            # (writer-only). Empty when no snapshot this row (raw geom absent).
            _m = getattr(self, "_raw_truth_meta", None)
            for _k in ("foll_tbms", "foll_wire", "foll_lat", "foll_lon", "foll_alt",
                       "foll_vn", "foll_ve", "foll_vd", "lead_tbms", "lead_lat",
                       "lead_lon", "lead_alt", "lead_vn", "lead_ve", "lead_hdg",
                       "lead_lat_al", "lead_lon_al", "age2_s", "home_lat",
                       "repoch_state", "repoch_ms"):
                row["rm_" + _k] = _m.get(_k) if _m is not None else ""
            self._paired_csv_writer.writerow(row)
            # T-392a: epoch-stamp witness -> its OWN file (paired CSV stays
            # byte-identical; see _ESTAMP_CSV_FIELDS). Never raises.
            self._write_estamp_row(row.get("sim_time_s"),
                                   row.get("episode_idx"),
                                   row.get("tick_idx"))
        except Exception:
            logger.debug("paired_channels.csv write failed", exc_info=True)

    # ------------------------------------------------------------------
    # Internal: observation & commands
    # ------------------------------------------------------------------

    def _t0sl_poison_truth(self) -> None:
        """PROBE mode only: overwrite the observability-only truth geom with a
        canary so any leak of truth into obs/reward/replay becomes detectable.
        Control-inert — _raw_truth_geom feeds paired logging only (env:~3513)."""
        g = self._raw_truth_geom
        if g is None:
            return
        g.behind_m = _T0SL_CANARY
        g.cross_m = _T0SL_CANARY
        g.below_m = _T0SL_CANARY
        g.range_m = _T0SL_CANARY

    def _t0sl_probe_step(self, obs: np.ndarray, reward: float) -> bool:
        """PROBE mode: track per-step obs/reward extremes; flag a canary leak."""
        self._t0sl_steps += 1
        mo = float(np.max(np.abs(obs))) if getattr(obs, "size", 0) else 0.0
        self._t0sl_max_obs = max(self._t0sl_max_obs, mo)
        self._t0sl_max_reward = max(self._t0sl_max_reward, abs(float(reward)))
        leak = _t0sl_leak(obs, reward)
        if leak:
            self._t0sl_leaks += 1
        return leak

    def _build_observation(
        self, dt_s: float, wake_hazard_intensity: float = 0.0,
    ) -> np.ndarray:
        """Build the 42-D observation from current state (config.OBS_DIM)."""
        if (
            self._follower_state is None
            or self._leader_state is None
            or self._last_geom is None
        ):
            return np.zeros(config.OBS_DIM, dtype=np.float32)

        geom = self._last_geom
        target_behind, target_below = self._get_phase_targets(geom)

        envelope = EnvelopeFlags(
            in_inner=self._envelope.is_in_precontact_inner_envelope(
                geom, self._leader_state.alt_msl_m, self._follower_state.alt_msl_m,
            ),
            in_outer=self._envelope.is_in_precontact_outer_envelope(
                geom, self._leader_state.alt_msl_m, self._follower_state.alt_msl_m,
            ),
        )
        sensor = SensorInfo(mode_id=1, uwb_valid=True)

        # Bearing from follower to leader (degrees, 0=North, CW positive)
        dlat = self._leader_lat - self._follower_lat
        dlon = self._leader_lon - self._follower_lon
        bearing_to_leader = math.degrees(math.atan2(dlon, dlat)) % 360.0

        return self._obs_builder.build(
            geom=geom,
            follower=self._follower_state,
            leader=self._leader_state,
            phase_id=self._get_phase_id(geom),
            envelope=envelope,
            sensor=sensor,
            dt_s=dt_s,
            continuous_contact_m3b_dwell_s=self._continuous_contact_m3b_dwell_s,
            milestone_tier=self._milestone_tier,
            target_behind=target_behind,
            target_below=target_below,
            speed_delta_ms=0.0,
            max_closure_ms=config.DEFAULT_MAX_CLOSURE_MS,
            bearing_to_leader_deg=bearing_to_leader,
            wake_hazard_intensity=wake_hazard_intensity,
        )

    def _apply_faultinj_vz(self, alt_cmd: float, dt_s: float) -> float:
        """#461 SIM-only fault injection: while in a
        docking-phase FSM state, ramp the commanded altitude UP at
        config.FAULTINJ_VZ_MS — a rogue full-authority climb toward the DHC-2
        belly, the exact hazard the #452 clamp must arrest. Applied UPSTREAM of
        _apply_belly_clamp so the clamp acts on the faulted command. Default-unset
        (0) ⇒ byte-identical no-op. The offset accumulates and applies ONLY
        in-phase (never fights a breakaway escape; SG-SAFE) and is capped at
        FAULTINJ_CAP_M so the sim cannot diverge. Reads FSM state + config only —
        no truth field (G4-clean). NEVER set in training/eval.
        """
        vz = config.FAULTINJ_VZ_MS
        if vz <= 0.0 or alt_cmd is None:
            return alt_cmd
        state_name = ""
        if self._ctrl_ref is not None:
            try:
                state_name = self._ctrl_ref.get_fsm_state().name
            except Exception:
                state_name = ""
        if state_name not in config.BELLY_CLAMP_STATES:
            return alt_cmd
        self._faultinj_alt_offset = min(
            self._faultinj_alt_offset + vz * dt_s, config.FAULTINJ_CAP_M)
        return alt_cmd + self._faultinj_alt_offset

    def _apply_belly_clamp(self, alt_cmd: float, dt_s: float) -> float:
        """#452 belly-collision Build-A: bound the UPWARD rate of the realized
        altitude command while in a docking-phase FSM state. Default-OFF ⇒
        byte-identical no-op. Descent is NEVER clamped (dive-away escape). Gate
        reads FSM state + the passed command only — no truth field (G4-clean).
        Tracks the sent command EVERY tick so envelope entry adds no step.
        """
        if not config.BELLY_CLAMP_ENABLE or alt_cmd is None:
            self._belly_clamp_diag = {
                "alt_cmd_pre_belly_clamp": alt_cmd if alt_cmd is not None else "",
                "alt_cmd_post_belly_clamp": alt_cmd if alt_cmd is not None else "",
                "belly_clamp_active": 0, "belly_clamp_limit": ""}
            return alt_cmd
        prev = self._belly_prev_alt_cmd
        state_name = ""
        if self._ctrl_ref is not None:
            try:
                state_name = self._ctrl_ref.get_fsm_state().name
            except Exception:
                state_name = ""
        pre_cmd = alt_cmd
        clamped = False
        limit = ""
        if state_name in config.BELLY_CLAMP_STATES and prev is not None:
            max_climb_step = config.BELLY_CLAMP_VZ_MS * dt_s   # UP toward the belly
            limit = prev + max_climb_step
            if alt_cmd - prev > max_climb_step:
                alt_cmd = prev + max_climb_step
                clamped = True
        self._belly_prev_alt_cmd = alt_cmd
        # OP-S3 (2026-07-26): observability capture ONLY — read by the step_end
        # diag emit. The clamp arithmetic above is byte-identical to pre-OP-S3.
        self._belly_clamp_diag = {
            "alt_cmd_pre_belly_clamp": pre_cmd,
            "alt_cmd_post_belly_clamp": alt_cmd,
            "belly_clamp_active": int(clamped),
            "belly_clamp_limit": limit}
        return alt_cmd

    def _send_commands(
        self, speed_ms: float, heading_deg: float, altitude_m: float,
    ) -> None:
        """Send DO_CHANGE_SPEED + DO_REPOSITION + altitude fallback via MAVLink.

        ArduPlane GUIDED mode navigates to waypoints, not headings directly.
        We compute a "carrot point" 1000m ahead in the desired heading direction
        and send DO_REPOSITION via COMMAND_INT (matching baseline controller).

        ArduPlane's TECS spreads the altitude change over the carrot distance,
        producing a negligible climb rate. The standalone controller compensates
        by also sending MAV_CMD_GUIDED_CHANGE_ALTITUDE on target changes and as
        a periodic fallback. We replicate that here.
        """
        if self._mav is None:
            return
        try:
            from pymavlink import mavutil as _mu

            now = getattr(self, '_time_auth', None)
            now_s = now.now_s() if now is not None else 0.0

            # Bounded target-drift detection + corrective re-pin.
            # See [ENV_TARGET_PIN] handoff log; if pymavlink mutates target
            # mid-flight (e.g., via heartbeat handler), re-pin to (1,1)
            # before sending. Logs first 5 drift events; counter exposed
            # in [ENV_MAVLINK_SUMMARY].
            if (self._mav.target_system != 1
                    or self._mav.target_component != 1):
                self._diag_target_drift_count = getattr(
                    self, '_diag_target_drift_count', 0) + 1
                if self._diag_target_drift_count <= 5:
                    logger.warning(
                        "[ENV_TARGET_DRIFT] sim=%.3f drift_count=%d "
                        "observed_sys=%d observed_comp=%d "
                        "(re-pinning to (1,1))",
                        now_s, self._diag_target_drift_count,
                        self._mav.target_system,
                        self._mav.target_component,
                    )
                self._mav.target_system = 1
                self._mav.target_component = 1

            if _FD_ENV_MAVLINK_DIAG:
                # SEND_TICK: ≤1Hz sim liveness summary
                last_tick_sim = getattr(self, '_diag_last_send_tick_log_sim', 0.0)
                if now_s - last_tick_sim >= 1.0:
                    self._diag_send_tick_count = getattr(self, '_diag_send_tick_count', 0) + 1
                    logger.warning("[ENV_SEND_TICK] sim=%.3f", now_s)
                    self._diag_last_send_tick_log_sim = now_s
                # SUMMARY: ≤0.2Hz sim, bounded counters
                last_sum_sim = getattr(self, '_diag_last_summary_log_sim', 0.0)
                if now_s - last_sum_sim >= 5.0:
                    logger.warning(
                        "[ENV_MAVLINK_SUMMARY] sim=%.3f send_ticks=%d "
                        "speed_sends=%d reposition_main_sends=%d "
                        "alt_target_change_sends=%d alt_fallback_sends=%d "
                        "exceptions=%d guided_heading_sends=%d "
                        "guided_heading_exceptions=%d "
                        "do_reposition_between_heading_sends_count_max=%d "
                        "target_drift_count=%d",
                        now_s,
                        getattr(self, '_diag_send_tick_count', 0),
                        getattr(self, '_diag_speed_send_count', 0),
                        getattr(self, '_diag_reposition_main_send_count', 0),
                        getattr(self, '_diag_alt_target_change_send_count', 0),
                        getattr(self, '_diag_alt_fallback_send_count', 0),
                        getattr(self, '_diag_exception_count', 0),
                        getattr(self, '_diag_guided_heading_send_count', 0),
                        getattr(self, '_diag_guided_heading_exception_count', 0),
                        getattr(self, '_diag_do_reposition_between_heading_sends_count_max', 0),
                        getattr(self, '_diag_target_drift_count', 0),
                    )
                    self._diag_last_summary_log_sim = now_s
                # ENV_HEADING_PERSISTENCE_CHECK at ≤0.5 Hz sim (every 2 sim-s)
                last_pcheck = getattr(self, '_diag_last_persistence_check_sim', 0.0)
                if now_s - last_pcheck >= 2.0:
                    last_hd = getattr(self, '_diag_last_guided_heading_send_sim', 0.0)
                    logger.warning(
                        "[ENV_HEADING_PERSISTENCE_CHECK] sim=%.3f "
                        "last_guided_heading_send_sim=%.3f "
                        "sim_since_last_guided_heading_send=%.3f "
                        "do_reposition_between_heading_sends_count=%d",
                        now_s, last_hd, max(0.0, now_s - last_hd),
                        getattr(self, '_diag_do_reposition_between_heading_sends_count', 0),
                    )
                    self._diag_last_persistence_check_sim = now_s

            # --- DO_CHANGE_SPEED: rate-limited (standalone: 0.10s / 10Hz) ---
            speed_interval = now_s - self._last_speed_send_time
            speed_delta = abs(speed_ms - self._last_speed_send_value)
            if (speed_interval >= config.CMD_SPEED_MIN_INTERVAL_S
                    or speed_delta > config.CMD_SPEED_DELTA_THRESHOLD):
                if _FD_ENV_MAVLINK_DIAG:
                    logger.warning(
                        "[ENV_CHANGE_SPEED_SEND] sim=%.3f tgt_sys=%d tgt_comp=%d "
                        "speed_ms=%.4f",
                        now_s, self._mav.target_system, self._mav.target_component,
                        speed_ms,
                    )
                self._mav.mav.command_long_send(
                    self._mav.target_system,
                    self._mav.target_component,
                    _mu.mavlink.MAV_CMD_DO_CHANGE_SPEED,
                    0,    # confirmation
                    0,    # speed type (0=airspeed)
                    speed_ms,
                    -1,   # throttle (-1 = no change)
                    0, 0, 0, 0,
                )
                self._last_speed_send_time = now_s
                self._last_speed_send_value = speed_ms
                if _FD_ENV_MAVLINK_DIAG:
                    self._diag_speed_send_count = getattr(self, '_diag_speed_send_count', 0) + 1

            # #329 Tranche-H4 single-writer: C++ owns lateral+alt while engaged
            # — env sends speed only (mirrors ctrl:5001-5013).
            if (_FD_DRLREF_POSTTICK_PARITY
                    and getattr(self._ctrl_ref, "_cpp_enabled", False)):
                return

            # --- DO_REPOSITION: rate-limited + altitude forcing ---
            # Standalone controller (lines 4640-4698) forces DO_REPOSITION on:
            # (a) altitude target change > 0.5m, (b) reposition alt delta > 0.5m,
            # (c) persistent altitude error > 5m for > 2s. Without these,
            # TECS spreads altitude changes over the 1000m carrot → negligible
            # climb rate → VERT_STABILIZE gate never passes → CREEP_CLOSE
            # never activates.
            reposition_interval = now_s - self._last_reposition_time
            hdg_delta = abs(heading_deg - self._last_reposition_hdg)
            if hdg_delta > 180:
                hdg_delta = 360 - hdg_delta

            # Altitude forcing checks (match standalone)
            force_alt_send = False
            alt_target_changed = False
            fol_alt = (
                self._follower_state.alt_msl_m
                if self._follower_state is not None else altitude_m
            )

            # (a) Altitude TARGET change (e.g., below_target ramp)
            if abs(altitude_m - self._last_alt_cmd_target) > config.CMD_ALT_TARGET_DELTA_M:
                alt_target_changed = True
                force_alt_send = True

            # (b) Reposition alt delta
            alt_delta = abs(altitude_m - self._last_reposition_alt)
            if not force_alt_send and alt_delta > config.CMD_ALT_TARGET_DELTA_M:
                force_alt_send = True

            # (c) Persistent altitude error
            alt_err = abs(fol_alt - altitude_m)
            if alt_err > config.CMD_ALT_ERR_FORCE_M:
                if self._alt_err_exceed_start is None:
                    self._alt_err_exceed_start = now_s
                elif (now_s - self._alt_err_exceed_start) >= config.CMD_ALT_ERR_FORCE_TIME_S:
                    force_alt_send = True
            else:
                self._alt_err_exceed_start = None

            # FIX-α (#328): forced sends re-arm on the interval limit (flag-gated)
            if force_alt_send and not force_send_allowed(
                    _FD_ENV_REPOSITION_FORCE_REARM, alt_target_changed, now_s,
                    self._last_reposition_time, config.CMD_REPOSITION_MIN_INTERVAL_S):
                force_alt_send = False

            should_send = (
                reposition_interval >= config.CMD_REPOSITION_MIN_INTERVAL_S
                or hdg_delta > config.CMD_REPOSITION_HDG_DELTA_DEG
                or force_alt_send
            )

            if should_send:
                # Compute carrot point 1000m ahead in the desired heading
                cur_lat = self._follower_lat
                cur_lon = self._follower_lon
                carrot_dist_m = config.CARROT_DISTANCE_M
                heading_rad = math.radians(heading_deg)

                meters_per_deg_lat = 111320.0
                meters_per_deg_lon = 111320.0 * math.cos(math.radians(cur_lat))

                delta_lat = (carrot_dist_m * math.cos(heading_rad)) / meters_per_deg_lat
                delta_lon = (carrot_dist_m * math.sin(heading_rad)) / meters_per_deg_lon

                target_lat_i = int((cur_lat + delta_lat) * 1e7)
                target_lon_i = int((cur_lon + delta_lon) * 1e7)

                if _FD_ENV_MAVLINK_DIAG:
                    logger.warning(
                        "[ENV_REPOSITION_SEND] sim=%.3f tgt_sys=%d tgt_comp=%d "
                        "command=MAV_CMD_DO_REPOSITION hdg_cmd=%.3f alt_cmd=%.3f "
                        "target_lat_i=%d target_lon_i=%d branch=reposition_main",
                        now_s, self._mav.target_system, self._mav.target_component,
                        heading_deg, altitude_m, target_lat_i, target_lon_i,
                    )
                self._mav.mav.command_int_send(
                    self._mav.target_system,
                    self._mav.target_component,
                    _mu.mavlink.MAV_FRAME_GLOBAL_INT,
                    _mu.mavlink.MAV_CMD_DO_REPOSITION,
                    0, 0,  # current, autocontinue
                    -1,    # param1: ground speed (-1 = use current)
                    1,     # param2: MAV_DO_REPOSITION_FLAGS_CHANGE_MODE
                    0,     # param3: reserved
                    heading_deg,  # param4: yaw heading in degrees
                    target_lat_i,  # x: latitude (int, 1e7 scaled)
                    target_lon_i,  # y: longitude (int, 1e7 scaled)
                    altitude_m,    # z: altitude MSL
                )
                self._last_reposition_time = now_s
                self._last_reposition_hdg = heading_deg
                self._last_reposition_alt = altitude_m
                if _FD_ENV_MAVLINK_DIAG:
                    self._diag_reposition_main_send_count = getattr(
                        self, '_diag_reposition_main_send_count', 0) + 1
                    # H5 fix: track DO_REPOSITION events between HEADING sends
                    self._diag_do_reposition_between_heading_sends_count = getattr(
                        self, '_diag_do_reposition_between_heading_sends_count', 0) + 1
                    cur = self._diag_do_reposition_between_heading_sends_count
                    if cur > getattr(self, '_diag_do_reposition_between_heading_sends_count_max', 0):
                        self._diag_do_reposition_between_heading_sends_count_max = cur

                # H5 fix: emit GUIDED_CHANGE_HEADING immediately after DO_REPOSITION
                # to re-arm ArduPlane's heading PID. DO_REPOSITION clears
                # target_heading_type to GUIDED_HEADING_NONE per
                # GCS_MAVLink_Plane.cpp:629, so a follow-up GUIDED_CHANGE_HEADING
                # is required to engage the FW heading PID at mode_guided.cpp:45-73.
                # #331: withhold this re-arm in DC states (FD_ENV_DC_L1_NAV) so
                # target_heading_type stays NONE -> L1 cross-track nav owns lateral.
                _dc_suppress_heading = (
                    _FD_ENV_DC_L1_NAV
                    and self._ctrl_ref is not None
                    and getattr(
                        getattr(self._ctrl_ref, "state", None), "name", None
                    ) in _DC_L1_STATES
                )
                if _dc_suppress_heading:
                    if _FD_ENV_MAVLINK_DIAG:
                        logger.warning(
                            "[ENV_DC_L1_NAV] sim=%.3f state=%s "
                            "branch=guided_heading_suppressed_dc",
                            now_s,
                            getattr(getattr(self._ctrl_ref, "state", None),
                                    "name", "?"),
                        )
                else:
                    heading_deg_norm = heading_deg % 360.0
                    if _FD_ENV_MAVLINK_DIAG:
                        last_dr = self._last_reposition_time
                        logger.warning(
                            "[ENV_GUIDED_HEADING_SEND] sim=%.3f tgt_sys=%d tgt_comp=%d "
                            "command=MAV_CMD_GUIDED_CHANGE_HEADING param1_heading_type=%d "
                            "param2_heading_deg_norm=%.3f param3_accel_limit=%.3f "
                            "sim_since_last_do_reposition_s=%.3f "
                            "branch=guided_heading_after_do_reposition",
                            now_s, self._mav.target_system, self._mav.target_component,
                            _mu.mavlink.HEADING_TYPE_HEADING, heading_deg_norm, 0.0,
                            max(0.0, now_s - last_dr),
                        )
                    try:
                        self._mav.mav.command_int_send(
                            self._mav.target_system,
                            self._mav.target_component,
                            _mu.mavlink.MAV_FRAME_GLOBAL,    # matches sister at line 2668
                            _mu.mavlink.MAV_CMD_GUIDED_CHANGE_HEADING,
                            0, 0,    # current, autocontinue
                            _mu.mavlink.HEADING_TYPE_HEADING,  # param1 = 1
                            heading_deg_norm,                  # param2 in [0, 360)
                            0.0,     # param3 (no slew limit; ArduPlane floors to 0.05 rad/s²)
                            0.0,     # param4 unused
                            0,       # x unused
                            0,       # y unused
                            0.0,     # z unused
                        )
                        self._diag_last_guided_heading_send_sim = now_s
                        if _FD_ENV_MAVLINK_DIAG:
                            self._diag_guided_heading_send_count = getattr(
                                self, '_diag_guided_heading_send_count', 0) + 1
                            self._diag_do_reposition_between_heading_sends_count = 0
                            logger.warning(
                                "[ENV_GUIDED_HEADING_SENT] sim=%.3f "
                                "branch=guided_heading_after_do_reposition",
                                now_s,
                            )
                    except Exception as _hdg_exc:
                        if _FD_ENV_MAVLINK_DIAG:
                            self._diag_guided_heading_exception_count = getattr(
                                self, '_diag_guided_heading_exception_count', 0) + 1
                            logger.warning(
                                "[ENV_GUIDED_HEADING_EXCEPTION] sim=%.3f "
                                "exc_class=%s exc_msg=%s",
                                now_s, type(_hdg_exc).__name__, str(_hdg_exc),
                                exc_info=True,
                            )

            # On altitude target change, also fire GUIDED_CHANGE_ALTITUDE
            # immediately (standalone lines 4695-4698) to bypass TECS lag
            if alt_target_changed:
                if _FD_ENV_MAVLINK_DIAG:
                    logger.warning(
                        "[ENV_REPOSITION_SEND] sim=%.3f tgt_sys=%d tgt_comp=%d "
                        "command=MAV_CMD_GUIDED_CHANGE_ALTITUDE alt_cmd=%.3f "
                        "branch=alt_target_change",
                        now_s, self._mav.target_system, self._mav.target_component,
                        altitude_m,
                    )
                self._mav.mav.command_int_send(
                    self._mav.target_system,
                    self._mav.target_component,
                    _mu.mavlink.MAV_FRAME_GLOBAL,
                    _mu.mavlink.MAV_CMD_GUIDED_CHANGE_ALTITUDE,
                    0, 0,  # current, autocontinue
                    0.0,   # param1: empty
                    0.0,   # param2: empty
                    0.0,   # param3: rate m/s (0 = max rate)
                    0.0,   # param4: empty
                    0, 0,  # x, y: lat/lon (0 = no change)
                    altitude_m,  # z: altitude MSL
                )
                self._last_alt_cmd_target = altitude_m
                self._last_guided_alt_target = altitude_m
                self._last_guided_alt_time = now_s
                if _FD_ENV_MAVLINK_DIAG:
                    self._diag_alt_target_change_send_count = getattr(
                        self, '_diag_alt_target_change_send_count', 0) + 1

            # --- GUIDED_CHANGE_ALTITUDE: error-based periodic fallback ---
            # When altitude error exceeds threshold and enough time has elapsed,
            # send GUIDED_CHANGE_ALTITUDE to force TECS to re-acquire target.
            fallback_due = (
                alt_err > config.CMD_ALT_ERR_FORCE_M
                and (now_s - self._last_guided_alt_time) >= 2.5
                and not alt_target_changed  # already sent above
            )

            if fallback_due:
                if _FD_ENV_MAVLINK_DIAG:
                    logger.warning(
                        "[ENV_REPOSITION_SEND] sim=%.3f tgt_sys=%d tgt_comp=%d "
                        "command=MAV_CMD_GUIDED_CHANGE_ALTITUDE alt_cmd=%.3f "
                        "branch=alt_fallback",
                        now_s, self._mav.target_system, self._mav.target_component,
                        altitude_m,
                    )
                self._mav.mav.command_int_send(
                    self._mav.target_system,
                    self._mav.target_component,
                    _mu.mavlink.MAV_FRAME_GLOBAL,
                    _mu.mavlink.MAV_CMD_GUIDED_CHANGE_ALTITUDE,
                    0, 0,  # current, autocontinue
                    0.0,   # param1: empty
                    0.0,   # param2: empty
                    0.0,   # param3: rate m/s (0 = max rate)
                    0.0,   # param4: empty
                    0, 0,  # x, y: lat/lon (0 = no change)
                    altitude_m,  # z: altitude MSL
                )
                self._last_guided_alt_target = altitude_m
                self._last_guided_alt_time = now_s
                if _FD_ENV_MAVLINK_DIAG:
                    self._diag_alt_fallback_send_count = getattr(
                        self, '_diag_alt_fallback_send_count', 0) + 1

        except Exception as _exc:
            if _FD_ENV_MAVLINK_DIAG:
                _now_for_log = locals().get('now_s', 0.0)
                self._diag_exception_count = getattr(self, '_diag_exception_count', 0) + 1
                logger.warning(
                    "[ENV_MAVLINK_EXCEPTION] sim=%.3f exc_class=%s exc_msg=%s",
                    _now_for_log, type(_exc).__name__, str(_exc),
                    exc_info=True,
                )
            else:
                logger.debug("Failed to send guided commands to SITL", exc_info=True)

    # ------------------------------------------------------------------
    # Internal: phase & termination
    # ------------------------------------------------------------------

    def _get_phase_targets(
        self, geom: RelativeGeometry,
    ) -> Tuple[float, float]:
        """Get current phase-dependent behind/below targets.

        Path 1: Dynamic targets from FSM state. The controller's FSM
        transitions through HOLD_PRECONTACT → DOCKING_CREEP → HOLD_CONTACT.
        Envelope errors (for reward and M3b checks) use the appropriate
        target for the current phase.

        Target values (boom-tip referenced):
          HOLD_PRECONTACT: 10m behind/below boom tip
          DOCKING_CREEP+: 0m behind/below boom tip (contact)
        """
        if self._ctrl_ref is not None:
            fsm_state = self._ctrl_ref.get_fsm_state()
            # Once the FSM enters DOCKING_CREEP or later, the reference
            # model is commanding approach to the boom tip (contact position).
            _APPROACH_STATES = {
                ControllerState.DOCKING_CREEP,
                ControllerState.HOLD_CONTACT,
                ControllerState.ALT_CAP_HOLD_RANGE,
                ControllerState.SUCCESS,
            }
            if fsm_state in _APPROACH_STATES:
                return fc.CONTACT_BEHIND_M, fc.CONTACT_BELOW_M
        return fc.PRECONTACT_BEHIND_M, fc.PRECONTACT_BELOW_M

    def _get_phase_id(self, geom: RelativeGeometry) -> int:
        """Map geometry to phase ID (0=INIT, 1=REJOIN, 2=PRECONTACT, 3=CONTACT)."""
        if geom.behind_m > 30.0:
            return 1  # REJOIN
        elif geom.behind_m > 5.0:
            return 2  # PRECONTACT
        else:
            return 3  # CONTACT

    def _check_m3b(
        self, behind_err: float, cross_err: float, below_err: float,
    ) -> bool:
        """Check if within tight precontact envelope (DRL 'hold_m3b').

        This checks ±0.7m of the PRECONTACT hold point (10m behind/below boom
        tip). It measures station-keeping quality, NOT boom-tip proximity.
        For true FlightDock M3b (boom-tip), see ``_check_contact_m3b()``.

        Body delegated to module-level :func:`check_m3b` (C3X Intervention 3
        refactor) so the ``QValueDistributionCallback`` can invoke the
        same membership logic on replay-buffer transitions without
        instantiating an env.
        """
        return check_m3b(behind_err, cross_err, below_err)

    # ------------------------------------------------------------------
    # v19: True FlightDock contact milestones (boom-tip reference)
    # ------------------------------------------------------------------
    # These use geom.behind_m / below_m / cross_m directly — the geometry
    # contract guarantees these are measured from the boom tip. Target is
    # implicitly (0, 0, 0). Precontact milestones above use *errors* from
    # the 10m standoff hold point — a different reference frame.

    def _check_contact_m3a(self, geom: RelativeGeometry) -> bool:
        """True FlightDock M3a: ±1.2m behind/below boom tip, ±1.0m cross-track."""
        return (
            abs(geom.behind_m) <= config.CONTACT_M3A_BEHIND_TOL
            and abs(geom.below_m) <= config.CONTACT_M3A_BELOW_TOL
            and abs(geom.cross_m) <= config.CONTACT_M3A_CROSS_TOL
        )

    def _check_contact_m3b(self, geom: RelativeGeometry) -> bool:
        """True FlightDock M3b: ±0.7m behind/below boom tip, ±0.5m cross-track.

        Body delegated to module-level :func:`check_contact_m3b` (C3X
        IC4 baseline rework) so the ``QValueDistributionCallback`` can
        invoke the same membership logic on replay-buffer transitions
        without instantiating an env. Byte-equivalent semantics.
        """
        return check_contact_m3b(geom.behind_m, geom.cross_m, geom.below_m)

    def _compute_milestone_tier(
        self, behind_err: float, cross_err: float, below_err: float,
    ) -> int:
        """Return highest precontact milestone tier from envelope errors.

        Tiers 1-4 measure station-keeping quality at the precontact hold point
        (10m behind/below boom tip), NOT boom-tip proximity. For true FlightDock
        milestones (boom-tip reference), see ``_check_contact_m3a/m3b()``.
        Tiers are nested — if you miss tier N, you can't hit tier N+1.
        """
        tier = 0
        for t in (1, 2, 3, 4):
            b_tol, c_tol, bl_tol = config.MILESTONE_THRESHOLDS[t]
            if abs(behind_err) <= b_tol and abs(cross_err) <= c_tol and abs(below_err) <= bl_tol:
                tier = t
            else:
                break
        return tier

    @staticmethod
    def _rsafe_wire_flag() -> bool:
        """T-438: FD_RSAFE_WIRE_ENABLE gate for the r_safe guard wiring
        (default OFF — flag-off behavior is byte-identical literals)."""
        return os.environ.get("FD_RSAFE_WIRE_ENABLE", "0").strip() == "1"

    @staticmethod
    def _ceiling_shape_flag() -> bool:
        """SHAPE-585: FD_CEILING_SHAPE_ENABLE gate for the endpoint ceiling
        shaping terms (default OFF — flag-off behavior is byte-identical)."""
        return os.environ.get("FD_CEILING_SHAPE_ENABLE", "0").strip() == "1"

    @staticmethod
    def _nopass_shape_flag() -> bool:
        """OVSHAPE-20260816: FD_NOPASS_SHAPE_ENABLE gate for the endpoint
        no-pass shaping terms (default OFF — flag-off is byte-identical)."""
        return os.environ.get("FD_NOPASS_SHAPE_ENABLE", "0").strip() == "1"

    def _apply_nopass_shape(self, total_reward, reward_components):
        """OVSHAPE-20260816 endpoint seam: ONE per-env-step no-pass shaping
        term on the same END-of-window perception geometry the observation was
        built from — bit-exactly recomputable from stored replay-buffer
        endpoints. G4: perception geometry only (self._last_geom), truth never
        enters. getattr defaults (not bare attrs): unit-test fixtures build
        this env via __new__ without __init__ (T-438 precedent); default False
        == flag-off == byte-identical. Returns the (possibly updated) total.
        §5 item 2 aggregation happens HERE, once per env step, because the
        sub-tick accumulator loop never sees this endpoint component."""
        if not getattr(self, "_nopass_shape_enable", False) \
                or getattr(self, "_last_geom", None) is None:
            return total_reward
        _curr = self._last_geom.behind_m
        _prev = getattr(self, "_nopass_prev_behind_m", None)
        if _prev is None:
            _prev = _curr
        _r = nopass_shaping_reward(_prev, _curr)
        total_reward += _r
        reward_components["r_nopass_shape"] = _r
        self._nopass_prev_behind_m = _curr
        _acc = getattr(self, "_reward_accumulators", None)
        if _acc is not None and "r_nopass_shape" in _acc:
            _acc["r_nopass_shape"] += _r
        if _r > getattr(self, "_nopass_shape_max", float('-inf')):
            self._nopass_shape_max = _r
        if _r < getattr(self, "_nopass_shape_min", float('inf')):
            self._nopass_shape_min = _r
        return total_reward

    def _rsafe_guard_fires_now(self) -> int:
        """T-438: current cumulative BREAKAWAY-guard fire count on the
        controller (alt-ceiling + hard-overtake; speed-cap EXCLUDED by the
        2026-08-11 election). PERCEPTION-side counters (G4-clean). The
        step loop differences two samples around tick_once() to get the
        per-sub-tick delta — never pass this cumulative value to reward."""
        c = self._ctrl_ref
        if c is None:
            return 0
        return (int(getattr(c, "_guard_altitude_ceiling_fire_count", 0))
                + int(getattr(c, "_guard_hard_overtake_fire_count", 0)))

    # -----------------------------------------------------------------------
    # TELEMFIX-771. HOW AGE IS OBTAINED WITHOUT TOUCHING R7.1.
    #
    # The controller's caches carry no receive timestamp, and
    # tests/baseline_formation_controller.py is FROZEN, so one cannot be added
    # there. But the cache holds the MESSAGE OBJECT, and a new message means a
    # NEW OBJECT. Comparing the cached object to the one seen last tick is
    # therefore an exact arrival detector, costing one reference per channel.
    #
    # Identity is by `is`, deliberately NOT id(): an id can be reused after the
    # object is freed, which would read as "no new message" when in fact the
    # old one was collected. Holding the reference also keeps the object alive,
    # which is what makes `is` safe here.
    # -----------------------------------------------------------------------
    _TELEM_CHANNELS = ("vfr", "pos", "att")

    def _telem_note(self, channel: str, msg) -> None:
        """Age 0 on a new object, else += one tick. Called once per refresh."""
        if msg is None:
            return
        seen = self._telem_seen.get(channel)
        if seen is not msg:
            self._telem_seen[channel] = msg
            self._telem_age_s[channel] = 0.0
        else:
            self._telem_age_s[channel] = (self._telem_age_s.get(channel, 0.0)
                                          + config.TICK_DT_S)

    def _telem_fresh(self, channel: str) -> bool:
        """True when this channel is young enough to make a safety decision on.

        NEVER SEEN reads NOT FRESH -- that is the pre-first-message window
        where the 24.7 / 0.0 literals are live, and a guard must not judge on
        a value the vehicle never sent.
        """
        if not config.TELEM_FRESHNESS_ENABLE:
            return True
        age = self._telem_age_s.get(channel)
        return age is not None and age <= config.TELEM_MAX_AGE_S

    def _telem_abstain(self, guard: str) -> bool:
        """Record that `guard` declined to judge this tick. Always True, so
        the caller reads as `if self._telem_abstain(...): <skip the check>`."""
        self._telem_abstains[guard] = self._telem_abstains.get(guard, 0) + 1
        return True

    # -----------------------------------------------------------------------
    # TERMDEMOTE-767.
    #
    # WHY THE RETURN SITES AND NOT A WRAPPER. _check_termination is a flat
    # sequence of early returns ending in the TIMEOUT check. A wrapper that
    # swallowed a demoted terminal and returned "no terminal" would SKIP that
    # TIMEOUT check and make the episode UNBOUNDED. Demotion is therefore
    # applied AT each demotable return site, which then FALLS THROUGH to the
    # next check -- a genuine SAFETY_RANGE or TIMEOUT on the same tick still
    # ends the episode exactly as it does today.
    #
    # WHY ONE RECORD PER TERMINAL PER LEG. A demoted collision stays true for
    # thousands of consecutive ticks; one record per tick would be a log, not
    # an event.
    #
    # WHAT IS DELIBERATELY NOT HERE: no peak, no apex, no admissibility
    # crossing, not even a unit conversion. All of those are metric
    # computation, which the env is forbidden to do. They are derived OFFLINE
    # from the TELEM-766-A flight-state record, which logs the underlying
    # channels on every undecimated tick. This method copies counters the env
    # ALREADY HOLDS and does arithmetic on none of them.
    #
    # `streak` is the caller's own violation counter. A NEW INTERVAL is
    # detected as `streak <= last_streak` -- the counter cleared and re-climbed.
    # Tick NUMBERS are deliberately not used: _check_termination is not reached
    # on every sub-tick (SITL stall paths return early), so a gap in tick
    # numbers is not evidence that the breach cleared, whereas a counter that
    # went backwards is.
    # -----------------------------------------------------------------------
    # Flat NUMERIC scalars only, on purpose. The eval harness builds ONE dict
    # that becomes both a CSV row and the results_eid JSON episode record, so a
    # nested event dict would break the CSV. And no boolean column: this
    # project writes bools as the literal strings "True"/"False", where pandas
    # dtype inference is fragile (see parse_helpers / the Phase D remediation).
    # `demoted_*_ticks > 0` IS the predicate and it is an int, so no consumer
    # has to coerce anything. Absent columns mean a PRE-TERMDEMOTE leg -- which
    # readers must not confuse with a present 0.
    _DEMOTE_FIELDS = (("SAFETY_COLLISION", "collision"),
                      ("SAFETY_AIRSPEED", "airspeed"))

    def _demote_info_fields(self) -> Dict[str, float]:
        """The per-episode demoted-event record, flattened for the harness."""
        out: Dict[str, float] = {}
        for terminal, short in self._DEMOTE_FIELDS:
            ev = (getattr(self, "_demoted_events", None) or {}).get(terminal)
            out["demoted_%s_ticks" % short] = int(ev["ticks"]) if ev else 0
            out["demoted_%s_intervals" % short] = (int(ev["intervals"])
                                                   if ev else 0)
            out["demoted_%s_first_s" % short] = (round(ev["first_sim_time_s"], 3)
                                                 if ev else -1.0)
            out["demoted_%s_last_s" % short] = (round(ev["last_sim_time_s"], 3)
                                                if ev else -1.0)
        # TELEMFIX-771: an abstention that nobody can count is indistinguishable
        # from a guard that never had anything to say.
        for guard in ("airspeed", "alt", "descent"):
            out["telem_abstain_%s" % guard] = int(
                (getattr(self, "_telem_abstains", None) or {}).get(guard, 0))
        return out

    def _demote(self, terminal: str, streak: int) -> bool:
        """Record a demoted terminal as an event. True == caller falls through.

        See the comment block above for why this is not a wrapper and why no
        derived quantity is computed here.
        """
        if terminal not in config.DEMOTE_TERMINALS:
            return False
        t = self._time_auth.now_s() - self._episode_start_sim_s
        ev = self._demoted_events.get(terminal)
        if ev is None:
            self._demoted_events[terminal] = {
                "terminal": terminal, "ticks": 1, "intervals": 1,
                "first_sim_time_s": t, "last_sim_time_s": t,
                "last_streak_ticks": streak,
            }
            return True
        ev["ticks"] += 1
        if streak <= ev["last_streak_ticks"]:
            ev["intervals"] += 1
        ev["last_streak_ticks"] = streak
        ev["last_sim_time_s"] = t
        return True

    def _check_termination(self) -> Tuple[bool, bool, str]:
        """Check episode termination conditions.

        Returns (terminated, truncated, terminal_type).

        Range safety uses a two-tier system (Approach F):
          - Below SOFT_RESET_RANGE_M: no penalty, no termination
          - Between SOFT_RESET_RANGE_M and stage safety range: progressive
            penalty (via _range_penalty, added to reward in step())
          - Above stage safety range: hard termination (SAFETY_RANGE)
        """
        # SUCCESS: M3b dwell >= threshold
        if self._accumulated_m3b_dwell_s >= config.SUCCESS_DWELL_S:
            return True, False, "SUCCESS"

        # SAFETY: range divergence (stage-dependent threshold)
        if self._last_geom is not None and self._last_geom.range_m > self._safety_range_m:
            import json as _json
            diag = {
                "event": "SAFETY_RANGE_TERMINATION",
                "env_id": self.env_id,
                "tick": self._tick_count,
                "sim_time_s": self._time_auth.now_s() - self._episode_start_sim_s,
                "absolute_sim_time_s": self._time_auth.now_s(),
                "wall_time_s": time.monotonic(),
                "range_for_safety": self._last_geom.range_m,
                "behind_m": self._last_geom.behind_m,
                "cross_m": self._last_geom.cross_m,
                "below_m": self._last_geom.below_m,
                "safety_range_limit": self._safety_range_m,
                "range_delta_from_prev_tick": (
                    self._last_geom.range_m - getattr(self, '_prev_range_m', 0.0)
                    if getattr(self, '_prev_range_m', 0.0) > 0 else None
                ),
                "leader_lat": self._leader_lat,
                "leader_lon": self._leader_lon,
                "follower_lat": self._follower_lat,
                "follower_lon": self._follower_lon,
                "follower_vn_ms": self._follower_vn,
                "follower_ve_ms": self._follower_ve,
                "leader_vn_ms": self._leader_vn,
                "leader_ve_ms": self._leader_ve,
                "follower_airspeed_ms": (
                    self._follower_state.airspeed_ms
                    if self._follower_state else None),
                "leader_airspeed_ms": (
                    self._leader_state.airspeed_ms
                    if self._leader_state else None),
                "follower_heading_deg": (
                    self._follower_state.heading_deg
                    if self._follower_state else None),
                "leader_heading_deg": (
                    self._leader_state.heading_deg
                    if self._leader_state else None),
                "pause_count": getattr(self, '_pause_count', 0),
                "unpause_count": getattr(self, '_unpause_count', 0),
            }
            logger.info(f"[DRL env_id={self.env_id}] SAFETY_RANGE_DIAG: {_json.dumps(diag)}")
            return True, False, "SAFETY_RANGE"

        # SAFETY: altitude floor (AGL)
        # TELEMFIX-771: alt_msl_m comes from the position cache, which is held
        # indefinitely. Judging a floor breach on a half-second-old altitude is
        # not a safety check, so an aged channel ABSTAINS and is COUNTED.
        if (self._home_alt_msl is not None and self._follower_state is not None
                and (self._telem_fresh("pos")
                     or not self._telem_abstain("alt"))):
            alt_agl = self._follower_state.alt_msl_m - self._home_alt_msl
            if alt_agl < config.SAFETY_ALT_FLOOR_AGL_M:
                return True, False, "SAFETY_ALT"

        # SAFETY: excessive descent rate (Vd > threshold, down-positive)
        # TELEMFIX-771: vd_ms shares the position cache with the floor above.
        if (self._follower_state is not None
                and (self._telem_fresh("pos")
                     or not self._telem_abstain("descent"))
                and self._follower_state.vd_ms
                > config.SAFETY_DESCENT_RATE_MAX_MS):
            return True, False, "SAFETY_DESCENT"

        # NOPASSTERM-770: the two terminals the operator terminates for, ahead
        # of every demotable one (most severe first). Same channel selection
        # as SAFETY_COLLISION below -- truth only under T-434, per-tick GPS
        # fallback -- so no new truth path opens (G4). Flat early returns, not
        # a wrapper, for the TIMEOUT reason TERMDEMOTE-767 documents. Default
        # OFF: with FD_SAFETY_TERMINALS unset this block is two False tests.
        if config.SAFETY_TERMINALS:
            st_geom = self._last_geom
            if (config.COLLISION_TRUTH_CHANNEL_ENABLE
                    and self._raw_truth_geom is not None):
                st_geom = self._raw_truth_geom
            if st_geom is not None:
                # Airframe contact: ONE tick. Past it the sim has no contact
                # physics; the 2.00 m bar is a zone violation (T1), not this.
                if ("SAFETY_CONTACT" in config.SAFETY_TERMINALS
                        and st_geom.below_m < config.CONTACT_PLANE_BELOW_M):
                    return True, False, "SAFETY_CONTACT"
                # Well in front of the tip AND on the leader's axis:
                # continuous dwell, reset on any tick back inside (same shape
                # as _collision_ticks). NOPASSGATE-781: the off-axis term is
                # what separates a pass from a LEAD_TURN frame rotation --
                # being "in front of the DHC-2" means being on its axis, and a
                # rotating frame sweeps that axis across an aircraft parked
                # 13 m off it. Deliberately NOT a range test: when the pass is
                # genuine, range IS the ahead-distance, so a range gate would
                # suppress exactly the deep pass it exists to catch.
                st_offaxis = math.sqrt(
                    st_geom.cross_m ** 2 + st_geom.below_m ** 2)
                if (st_geom.behind_m < -config.NOPASS_TERM_AHEAD_M
                        and st_offaxis < config.NOPASS_TERM_OFFAXIS_M):
                    self._nopass_ticks += 1
                else:
                    self._nopass_ticks = 0
                if ("SAFETY_NOPASS" in config.SAFETY_TERMINALS
                        and self._nopass_ticks
                        >= config.NOPASS_TERM_PERSIST_TICKS):
                    return True, False, "SAFETY_NOPASS"

        # v13 Commit 13: SAFETY: airspeed envelope violation (2s sustained)
        if self._follower_state is not None:
            tas = self._follower_state.airspeed_ms
            # TELEMFIX-771. A stale or NaN airspeed HOLDS the counter: it
            # neither counts toward a violation nor clears one in progress.
            #
            # WHY HOLD AND NOT RESET. The historical `else` below is a RESET,
            # and both NaN comparisons (`tas < MIN`, `tas > MAX`) evaluate
            # False -- so a single bad reading fell into the reset branch and
            # ERASED an excursion already 39 ticks deep. Holding is the only
            # option that neither invents a violation nor destroys one.
            #
            # `tas != tas` is the NaN test: NaN is the sole value unequal to
            # itself, and it needs no import and no isinstance guard.
            if not self._telem_fresh("vfr") or tas != tas:
                self._telem_abstain("airspeed")
            elif tas < config.TAS_MIN_SAFE or tas > config.TAS_MAX_SAFE:
                self._airspeed_violation_ticks += 1
            else:
                self._airspeed_violation_ticks = 0
            if self._airspeed_violation_ticks >= config.AIRSPEED_VIOLATION_TICKS:
                if not self._demote("SAFETY_AIRSPEED",
                                    self._airspeed_violation_ticks):
                    return True, False, "SAFETY_AIRSPEED"

        # v13 Commit 13: SAFETY: collision proximity (above leader in docking zone)
        # Sustained-tick filter: require COLLISION_SUSTAINED_TICKS consecutive
        # ticks in collision zone before triggering (filters single-tick transients
        # from altitude oscillation during CREEP_CLOSE approach).
        # T-434 (FD_COLLISION_TRUTH_CHANNEL, default OFF): with the flag ON the
        # predicate judges on the SIM_STATE truth geometry per the contract of
        # record ("Truth below_m governs COLLISION"), falling back per-tick to
        # the GPS geometry when truth is unavailable (held->stale is accepted;
        # None falls back) so a truth outage can never blind the terminal.
        # SAFETY_RANGE, the observation and the range penalty deliberately stay
        # on _last_geom — they are perception-side by design (G4).
        coll_geom = self._last_geom
        if (config.COLLISION_TRUTH_CHANNEL_ENABLE
                and self._raw_truth_geom is not None):
            coll_geom = self._raw_truth_geom
        if coll_geom is not None:
            cg_dist = math.sqrt(
                coll_geom.behind_m ** 2
                + coll_geom.cross_m ** 2
                + coll_geom.below_m ** 2
            )
            follower_above = coll_geom.below_m < config.COLLISION_ABOVE_BOOM_M
            in_docking_zone = (
                config.COLLISION_BEHIND_MIN
                <= coll_geom.behind_m
                <= config.COLLISION_BEHIND_MAX
            )
            if cg_dist < config.COLLISION_CG_DISTANCE_MIN and follower_above and in_docking_zone:
                self._collision_ticks += 1
            else:
                self._collision_ticks = 0
            if self._collision_ticks >= config.COLLISION_SUSTAINED_TICKS:
                if not self._demote("SAFETY_COLLISION",
                                    self._collision_ticks):
                    return True, False, "SAFETY_COLLISION"

        # W-A (FD_FORMATION_LOST_TERM, default OFF): task-scope termination.
        # Ordered AFTER the safety aborts (a genuine safety event on the same
        # tick still wins) and BEFORE timeout. SAFETY_ prefix deliberate: the
        # frozen reward's startswith("SAFETY_") branch prices it like the
        # aborts it pre-empts (reward_mrrl untouched).
        if self._check_formation_lost():
            return True, False, "SAFETY_FORMATION_LOST"

        # TIMEOUT: sim-time exceeded
        elapsed = self._time_auth.now_s() - self._episode_start_sim_s
        if elapsed >= self.episode_duration_s:
            return False, True, "TIMEOUT"

        return False, False, ""

    def _check_formation_lost(self) -> bool:
        """W-A predicate: FSM entered the far re-intercept chain post-arm.

        The latch arms on the first ARM_STATES sighting after reset (all
        measured W0 episodes open in HOLD_PRECONTACT), so warmup/startup
        residue can never fire. EMERGENCY_BREAKAWAY does not fire — only
        FORMATION_LOST_STATES (INTERCEPT) after the latch is armed.
        """
        if not config.FORMATION_LOST_TERM_ENABLE:
            return False
        if self._ctrl_ref is None:
            return False
        fsm = getattr(getattr(self._ctrl_ref, "state", None), "name", "")
        if not self._formation_lost_armed:
            if fsm in config.FORMATION_LOST_ARM_STATES:
                self._formation_lost_armed = True
            return False
        return fsm in config.FORMATION_LOST_STATES

    def _compute_range_penalty(self) -> float:
        """Quadratic penalty for range beyond SOFT_WARN_RANGE_M.

        Returns a non-positive value that should be added to the reward.
        The penalty ramps quadratically from 0 at SOFT_WARN_RANGE_M to
        -RANGE_PENALTY_QUADRATIC_MAX at HARD_RANGE_MAX_M.
        """
        if self._last_geom is None:
            return 0.0
        if self._last_geom.range_m <= config.SOFT_WARN_RANGE_M:
            return 0.0
        excess = (self._last_geom.range_m - config.SOFT_WARN_RANGE_M) / (
            config.HARD_RANGE_MAX_M - config.SOFT_WARN_RANGE_M
        )
        return -config.RANGE_PENALTY_QUADRATIC_MAX * excess ** 2

    def _tick_sleep(self, dt_s: float) -> None:
        """Sleep to pace at ~20Hz wall-clock / speedup."""
        wall_dt = dt_s / max(self.speedup, 0.5)
        time.sleep(max(0.001, wall_dt * 0.8))  # Slightly fast to avoid drift
