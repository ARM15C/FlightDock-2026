"""Multi-component MRRL reward function for residual DRL.

Thirteen components (v12 + C006 Phase 5 + Phase 7 + Campaign 2 + Campaign 3):
  R_reference — penalise deviation from baseline (applied residual)
  R_proximity — logarithmic anisotropic distance to M3b envelope centre
  R_dwell — reward sustained time inside M3b envelope (stage-gated)
  R_safety — penalise guard fires and safety violations
  R_smoothness — penalise jerk in applied residual
  R_wake — penalise proximity to wake vortex cores
  R_m3a_coarse — flat per-tick reward inside M3a coarse envelope
  R_m3a_fine — dwell-bonus reward inside M3a fine envelope
  R_approach — forward-proximity bonus toward boom tip (C006 Phase 5)
  R_terminal — success/fail/timeout/safety terminal reward (unweighted)
  R_passthrough — anti-passthrough penalty for near-zero actions (C006 Phase 7)
  R_contact_dwell — boom-tip M3b dwell ramp (Campaign 2)
  R_action_rate — penalise sum-of-squares action rate (Campaign 3 P1-A)

All weights are stage-dependent (curriculum progression).
R_terminal, R_approach, and R_passthrough are unweighted.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from . import config


def _sigmoid_velocity_gate(v_rel_fwd: float, k: float = 15.0) -> float:
    """Sigmoid velocity gate for directional barrier reward.

    Returns modulation factor in [0, 1]:
      v >> 0 (forward): -> 1.0 (full barrier reward)
      v = 0 (perfect station-keep): = 0.5 (50% reward)
      v << 0 (braking/aft-drift): -> 0.0 at strong braking

    The 0.5 floor at v=0 emerges mathematically from sigmoid symmetry.
    No epsilon_floor needed — the raw sigmoid provides 50% reward at zero
    velocity (the council's asymmetric tanh formulation required an
    explicit floor; the symmetric sigmoid does not).
    """
    clamped = max(-10.0, min(10.0, k * v_rel_fwd))
    return 1.0 / (1.0 + math.exp(-clamped))


class MRRLRewardComputer:
    """Computes the 13-component MRRL reward (v12 + C006 + Campaign 2 + Campaign 3).

    Parameters
    ----------
    stage: int
        Initial curriculum stage (0-3).
    """

    # D6.1 Amendment-R: the PBRS constants and progress state are gone with
    # the >50 m branch — r_prox is stateless single-log for all distances.

    def __init__(self, stage: int = 0) -> None:
        self._weights = config.STAGE_WEIGHTS[stage]
        # CBF barrier state
        self._prev_behind_abs: float = 0.0
        self._prev_cross_abs: float = 0.0
        self._prev_below_abs: float = 0.0
        self._v_behind_ema: float = 0.0
        self._v_cross_ema: float = 0.0
        self._v_below_ema: float = 0.0

    def set_stage(self, stage: int) -> None:
        """Switch to new curriculum stage weights."""
        self._weights = config.STAGE_WEIGHTS[stage]

    @property
    def w_exit_penalty(self) -> float:
        """Effective exit-penalty weight for the current curriculum stage."""
        return self._weights.w_exit_penalty

    @property
    def w_overspeed(self) -> float:
        """Effective overspeed weight for the current curriculum stage."""
        return self._weights.w_overspeed

    def _compute_barrier_reward(
        self,
        behind_error_m: float,
        cross_error_m: float,
        below_error_m: float,
        in_m3b: bool,
        milestone_tier: int,
        dt_s: float,
    ) -> float:
        """CBF-style barrier reward — anticipatory gradient in terminal corridor.

        Phase-gated to milestone_tier >= 3 (M3a_fine+). C3o spec revision: original
        spec said tier >= 4 (M3b only); revised to provide gradient in M3a_fine->M3b
        approach corridor. Provides gradient APPROACHING and INSIDE M3b.
        No clipping at zero for violations — steep negative CBF penalty outside.
        """
        if milestone_tier < 3:
            self._prev_behind_abs = abs(behind_error_m)
            self._prev_cross_abs = abs(cross_error_m)
            self._prev_below_abs = abs(below_error_m)
            return 0.0

        abs_b = abs(behind_error_m)
        abs_c = abs(cross_error_m)
        abs_l = abs(below_error_m)

        # h(x) = boundary - |error|; positive = inside M3b, negative = outside
        h_b = config.CONTACT_M3B_BEHIND_TOL - abs_b
        h_c = config.CONTACT_M3B_CROSS_TOL - abs_c
        h_l = config.CONTACT_M3B_BELOW_TOL - abs_l

        # Proximity gate in ERROR SPACE: per-axis, activate when abs_error <= boundary * gate
        gate = config.BARRIER_PROXIMITY_GATE
        active_b = abs_b <= config.CONTACT_M3B_BEHIND_TOL * gate
        active_c = abs_c <= config.CONTACT_M3B_CROSS_TOL * gate
        active_l = abs_l <= config.CONTACT_M3B_BELOW_TOL * gate

        if not (active_b or active_c or active_l):
            self._prev_behind_abs = abs_b
            self._prev_cross_abs = abs_c
            self._prev_below_abs = abs_l
            return 0.0

        # EMA-filtered velocity toward boundary (reduces finite-diff noise at ~8Hz sim)
        alpha = config.BARRIER_VELOCITY_EMA
        self._v_behind_ema = (alpha * ((abs_b - self._prev_behind_abs) / max(dt_s, 0.001))
                              + (1.0 - alpha) * self._v_behind_ema)
        self._v_cross_ema = (alpha * ((abs_c - self._prev_cross_abs) / max(dt_s, 0.001))
                             + (1.0 - alpha) * self._v_cross_ema)
        self._v_below_ema = (alpha * ((abs_l - self._prev_below_abs) / max(dt_s, 0.001))
                             + (1.0 - alpha) * self._v_below_ema)

        v_b = max(0.0, self._v_behind_ema)
        v_c = max(0.0, self._v_cross_ema)
        v_l = max(0.0, self._v_below_ema)

        self._prev_behind_abs = abs_b
        self._prev_cross_abs = abs_c
        self._prev_below_abs = abs_l

        def _axis(h: float, v: float, v_max: float, active: bool) -> float:
            if not active:
                return 0.0
            if h > 0:
                # Inside envelope: positive reward, velocity-modulated
                # v > v_max -> reward goes NEGATIVE (actionable gradient)
                return h * (1.0 - (v / v_max) ** 2)
            else:
                # Outside envelope: steep negative CBF penalty
                return config.CBF_VIOLATION_SCALE * h * (1.0 + v / v_max)

        r_b = _axis(h_b, v_b, 2.0, active_b)
        r_c = _axis(h_c, v_c, 1.0, active_c)
        r_l = _axis(h_l, v_l, 1.0, active_l)

        # Asymmetric: behind 3x (bandwidth-limited axis)
        return 3.0 * r_b + 1.0 * r_c + 1.0 * r_l

    def compute(
        self,
        behind_error_m: float,
        cross_error_m: float,
        below_error_m: float,
        applied_residual: np.ndarray,      # shape (3,) — post-clamp
        prev_applied_residual: np.ndarray,  # shape (3,) — previous tick
        in_m3b: bool,
        continuous_dwell_s: float,
        guard_fires: int,
        safety_violations: int,
        is_terminal: bool,
        # Terminal names of record live in the env, NOT here. This line
        # listed "SUCCESS"/"FAIL"/"TIMEOUT"/"" and had gone stale: real
        # legs also emit SAFETY_RANGE, SAFETY_COLLISION, SAFETY_AIRSPEED,
        # STALL_SIMTIME and kin. A duplicated list goes stale again, so
        # this now points at the producer and copies nothing.
        terminal_type: str,                 # see env _check_termination
        wake_hazard_intensity: float = 0.0,
        in_m3a_coarse: bool = False,
        in_m3a_fine: bool = False,
        continuous_m3a_fine_dwell_s: float = 0.0,
        boom_tip_range_m: float | None = None,
        curriculum_stage: int = 0,
        raw_action: np.ndarray | None = None,
        in_contact_m3b: bool = False,
        continuous_contact_m3b_dwell_s: float = 0.0,
        dt_s: float = 0.05,
        milestone_tier: int = 0,
        closure_rate_ms: float = 0.0,  # follower forward velocity rel leader
        policy_speed_delta: float = 0.0,  # physical PSD in m/s (post action_scale)
    ) -> Tuple[float, Dict[str, float]]:
        """Compute weighted reward and per-component breakdown.

        Returns
        -------
        total: float
            Weighted sum of all components.
        components: dict
            Individual (unweighted) component values for logging.
        """
        w = self._weights

        # --- R_reference: penalise applied residual magnitude ---
        # Scale-aware: normalise each axis by its ACTION_SCALE before averaging.
        # ACTION_DIM=3 (cross-axis restored 2026-05-06; supersedes C3X Intervention 1):
        # applied_residual is (3,) with all axes live; ACTION_SCALES is 3-element
        # [speed, heading, altitude] (actuator) or [behind, cross, below] (reference).
        # Patch D 2026-05-06: index mapping [0, 1, 2] → [0, 1, 2]; 3-axis average;
        # heading axis [1] now penalized (was structurally ignored).
        a = np.asarray(applied_residual, dtype=np.float64)
        _ref_scales = (config.ACTION_SCALES_REFERENCE
                       if config.RESIDUAL_MODE == "reference"
                       else config.ACTION_SCALES)
        # Each axis normalised by its own scale; clamp scale to >=1e-8 to avoid div-by-zero.
        speed_norm = abs(a[0]) / max(_ref_scales[0], 1e-8)
        cross_norm = abs(a[1]) / max(_ref_scales[1], 1e-8)
        alt_norm = abs(a[2]) / max(_ref_scales[2], 1e-8)
        r_ref = -(speed_norm + cross_norm + alt_norm) / 3.0

        # --- R_proximity: single-log for ALL distances (D6.1 Amendment-R,
        # SIGNED 2026-07-18): the piecewise >50 m PBRS/progress branch is
        # DELETED (warmup-transient only per the signed census; it was NOT
        # strict PBRS — correction recorded in the memo). Stateless and
        # continuous by construction.
        d_aniso = math.sqrt(
            (behind_error_m / config.PROX_BEHIND_SCALE) ** 2
            + (cross_error_m / config.PROX_CROSS_SCALE) ** 2
            + (below_error_m / config.PROX_BELOW_SCALE) ** 2
        )
        r_prox = -math.log10(1.0 + d_aniso)

        # C006 Phase 3: r_prox decay inside M2 envelope
        # In the log branch (boom-tip range < 15m always coincides with
        # d_aniso <= 50m), r_prox <= 0, so multiplying by 0.2 makes it less
        # negative (less punishing). The PBRS branch (d_aniso > 50m) with
        # boom_tip_range < 15m is geometrically impossible in normal episodes,
        # so the decay is effectively log-branch-only.
        # Fixes the 182:1 punishment-to-reward ratio from Phase 2.
        if (boom_tip_range_m is not None
                and boom_tip_range_m < config.R_PROX_M2_RANGE_THRESHOLD_M):
            r_prox *= config.R_PROX_M2_DECAY

        # --- Campaign 3: envelope proximity sigmoid (P0-A + P0-B) ---
        if config.ZONE_BASED_WREF_ENABLED and boom_tip_range_m is not None:
            _exp = config.W_REF_SIGMOID_STEEPNESS * (
                boom_tip_range_m - config.W_REF_SIGMOID_CENTER_M)
            # Overflow guard: prevent math.exp explosion at large ranges
            envelope_proximity = 1.0 / (1.0 + math.exp(min(_exp, 500.0)))
            # P0-A: state-dependent w_ref
            w_ref_effective = (config.W_REF_OUTER
                               + (config.W_REF_INNER - config.W_REF_OUTER)
                               * envelope_proximity)
            # P0-B: suppress r_prox inside envelope
            prox_suppression = (1.0 - (1.0 - config.R_PROX_ENVELOPE_SUPPRESSION)
                                * envelope_proximity)
            w_prox_effective = w.w_prox * prox_suppression
        else:
            envelope_proximity = 0.0
            w_ref_effective = w.w_ref
            w_prox_effective = w.w_prox

        # --- R_dwell: stage-gated M3b dwell reward (v12, capped v13) ---
        dwell_threshold = config.DWELL_REWARD_THRESHOLD_S.get(curriculum_stage, 0.5)
        if in_m3b and continuous_dwell_s >= dwell_threshold:
            r_dwell = min(
                1.0 * (1.0 + config.R_DWELL_COEFF * continuous_dwell_s),
                config.DWELL_CAP_M3B,
            )
        else:
            r_dwell = 0.0

        # --- R_precontact: precontact tier-4 station-keeping pay (R1ELECT-703) ---
        # AMENDMENT R1 (SIGNED): r_dwell paid in only 2 of 63 measured blocks
        # while precontact_tier4_dwell_s reached 15.1 s -- the policy holds a
        # good precontact station and is paid nothing for it. This term pays
        # that station, continuously, so the operating point can move.
        #
        # Frame: the ERROR axes, which SWITCH anchor with FSM state (standoff
        # in HOLD_PRECONTACT and earlier, boom tip from DOCKING_CREEP on) --
        # see the deep-dive note on precontact_reward for the full finding.
        # Band (R1b, SIGNED 2026-08-19): TIERS 1-4 -- n normalized by the
        # TIER-1 tolerances, so the linear term spans the whole precontact
        # ladder instead of only the innermost band the policy merely buzzes.
        # Zero at the tier-1 outer edge => continuous across the boundary.
        # Hard zero inside CONTACT M3b (boom-tip frame): past contact this term
        # must not compete with r_contact_dwell, and must never pay for hanging
        # back. NOT zeroed on precontact `in_m3b` -- MEASURED: check_m3b() tests
        # the identical +-0.7/0.5/0.7 envelope, so `in_m3b` IS tier 4 and zeroing
        # on it would make this term identically 0.0 (a dead lever).
        #
        # Overlap with r_dwell is intended and the two are ORTHOGONAL:
        #   r_dwell pays for TIME held, flat in position, and only after
        #                continuous_dwell_s clears DWELL_REWARD_THRESHOLD_S.
        #   r_precontact pays for POSITION, flat in time, from the first tick.
        # That is the measured gap: tier-4 residence reached 15.1 s cumulative
        # while r_dwell paid in 2 of 63 blocks -- i.e. the policy crosses the
        # envelope repeatedly and rarely holds it CONTINUOUSLY past threshold.
        # This term's gradient (d r/d n = -1 everywhere in band) pulls inward
        # even at the edge where its VALUE is 0.
        # R1ARM-723: the body moved to the module-level `precontact_reward`
        # so the OFFLINE BUFFER REPRICE calls the identical function rather
        # than a copy of this formula. Same one-source-of-truth rule the
        # ceiling/nopass endpoint terms already follow; a re-implementation in
        # tools/ could drift from this call site without any test noticing.
        r_precontact = precontact_reward(
            behind_error_m, cross_error_m, below_error_m, in_contact_m3b)

        # --- R_contact_dwell: boom-tip M3b dwell ramp ---
        # Campaign 3c: power-law ramp eliminates buzzing attractor.
        # Linear (k=1) rewarded transient entries proportionally; power-law
        # (k=2) makes sustained dwell disproportionately more valuable.
        # See C3B_HALT_RECORD.md for buzzing attractor diagnosis.
        if in_contact_m3b and continuous_contact_m3b_dwell_s > 0.0:
            linear_ramp = min(
                continuous_contact_m3b_dwell_s / config.CONTACT_M3B_DWELL_S,
                config.CONTACT_DWELL_RAMP_CAP,
            )
            quadratic = linear_ramp ** config.CONTACT_DWELL_EXPONENT
            # linear ramp floor for dense early-dwell gradient.
            # Provides visible reward during first ~1.2s where quadratic is near-zero.
            ramp_floor = config.DWELL_RAMP_FLOOR * min(
                continuous_contact_m3b_dwell_s / 1.5, 1.0)
            r_contact_dwell = max(quadratic, ramp_floor)
        else:
            r_contact_dwell = 0.0

        # D6.1 Amendment-R (SIGNED 2026-07-18): r_m3a_coarse, r_m3a_fine,
        # r_action_rate, r_approach, r_passthrough TRIMMED (15 -> 10 terms;
        # gradients duplicated by survivors per the signed decomposition).

        # --- R_safety ---
        r_safe = guard_fires * (-1.0) + safety_violations * (-10.0)

        # --- R_smoothness: -||applied(t) - applied(t-1)||₂ ---
        diff = np.asarray(applied_residual, dtype=np.float64) - np.asarray(
            prev_applied_residual, dtype=np.float64
        )
        r_smooth = -float(np.linalg.norm(diff))

        # --- R_wake: penalise proximity to vortex cores ---
        r_wake = -wake_hazard_intensity

        # --- R_vsat: A2A AMENDMENT 2 (SIGNED 2026-09-01) -------------------
        # Anti-SATURATION penalty on the VERTICAL residual. It does NOT ask
        # the policy to push down less -- it asks it to stop pushing down AT
        # THE STOP. Quadratic, one-sided, zero inside the deadband, symmetric
        # in |.| so a saturated UP command (the H1 hazard direction) is
        # discouraged too.
        #
        # READS THE POLICY'S REQUEST, NOT THE APPLIED VALUE. `raw_action` is
        # the same quantity the flight record logs as `res_policy_2` (env
        # step: `_last_raw_action` / `_reslog_policy_action`, both from
        # `action`), which is what the band measurement was taken on. Scoring
        # the post-clamp `applied_residual` would penalise a different
        # quantity than the one measured. `raw_action` is captured AFTER the
        # inject-zero-action override, so an identity-proof leg scores 0.0.
        #
        # SHIPS INERT: w_vsat defaults to 0.0 on every stage of every ladder,
        # so this contributes exactly 0.0 and the total is byte-identical to
        # the pre-amendment baseline. Arming is a separate
        # (see config.W_VSAT_A2A). Inertness is the rollback AND the negative
        # control in tests/unit/test_vsat_term.py.
        r_vsat = 0.0
        if w.w_vsat and raw_action is not None and len(raw_action) > 2:
            _over = abs(float(raw_action[2])) - config.VSAT_DEADBAND
            if _over > 0.0:
                r_vsat = -(_over ** 2)

        # --- R_terminal ---
        if is_terminal:
            if terminal_type == "SUCCESS":
                r_terminal = config.R_TERMINAL_SUCCESS
            elif terminal_type in ("FAIL", "ABORT", "NO_PASS"):
                r_terminal = config.R_TERMINAL_FAIL
            elif terminal_type.startswith("SAFETY_"):
                r_terminal = config.R_TERMINAL_SAFETY
            else:
                r_terminal = config.R_TERMINAL_TIMEOUT
        else:
            r_terminal = 0.0

        # --- CBF barrier reward with velocity gate ---
        r_barrier = self._compute_barrier_reward(
            behind_error_m, cross_error_m, below_error_m,
            in_m3b, milestone_tier, dt_s)
        # Sigmoid velocity gate — inside precontact hold zone (in_m3b)
        if in_m3b:
            if config.RESIDUAL_MODE == "reference":
                # Gate disabled. closure_rate_ms is PID-dominated in reference
                # mode — saturates at >0.999 regardless of position offsets.
                # Post-mortem: vmod=1.0000 all 24 episodes. D1-D4 confirmed.
                velocity_mod = 1.0
            else:
                # Actuator: One-sided PSD gate on policy_speed_delta.
                velocity_mod = min(
                    1.0, 2.0 * _sigmoid_velocity_gate(policy_speed_delta, k=config.VELOCITY_GATE_K)
                )
            r_barrier *= velocity_mod
            r_dwell *= velocity_mod
        else:
            velocity_mod = 1.0  # No gate outside precontact hold zone

        # Additive braking penalty inside in_m3b.
        # Negative when policy brakes (PSD < 0), zero when forward/stationary.
        if in_m3b:
            r_hold_fwd = config.W_HOLD_FWD * min(policy_speed_delta, 0.0)
        else:
            r_hold_fwd = 0.0

        barrier_weight = config.W_BARRIER if in_m3b else config.W_BARRIER_OUTSIDE

        # --- Weighted sum (D6.1: 10 survivor terms; trimmed terms removed) ---
        total = (
            w_ref_effective * r_ref            # P0-A: was w.w_ref
            + w_prox_effective * r_prox        # P0-B: was w.w_prox
            + w.w_dwell * r_dwell
            + w.w_safe * r_safe
            + w.w_smooth * r_smooth
            + w.w_wake * r_wake
            + w.w_contact_dwell * r_contact_dwell
            + w.w_precontact * r_precontact
            + w.w_vsat * r_vsat  # A2A (ships INERT at 0.0)
            + r_terminal  # terminal is unweighted
            + barrier_weight * r_barrier  # CBF barrier (0 when W_BARRIER=0)
            + r_hold_fwd  # Additive braking penalty (unweighted, self-scaled)
        )

        components = {
            "r_ref": r_ref,
            "r_prox": r_prox,
            "r_dwell": r_dwell,
            "r_contact_dwell": r_contact_dwell,
            "r_precontact": r_precontact,
            "r_vsat": r_vsat,
            "r_safe": r_safe,
            "r_smooth": r_smooth,
            "r_wake": r_wake,
            "r_terminal": r_terminal,
            "r_barrier": r_barrier,
            "r_hold_fwd": r_hold_fwd,
            "velocity_mod": velocity_mod,
            "envelope_proximity": envelope_proximity,
            "d_aniso": d_aniso,
            "boom_tip_range_m": boom_tip_range_m if boom_tip_range_m is not None else -1.0,
            "total": total,
            "r_ref_weighted": w_ref_effective * r_ref,
            "r_prox_weighted": w_prox_effective * r_prox,
            "r_dwell_weighted": w.w_dwell * r_dwell,
            "r_safe_weighted": w.w_safe * r_safe,
            "r_smooth_weighted": w.w_smooth * r_smooth,
            "r_wake_weighted": w.w_wake * r_wake,
            "r_contact_dwell_weighted": w.w_contact_dwell * r_contact_dwell,
            "r_precontact_weighted": w.w_precontact * r_precontact,
            "r_vsat_weighted": w.w_vsat * r_vsat,
            "r_terminal_weighted": r_terminal,
            "r_barrier_weighted": barrier_weight * r_barrier,
            "r_hold_fwd_weighted": r_hold_fwd,
        }

        return total, components


# --- SHAPE-585: endpoint ceiling-shaping terms (module-level, STATELESS) ---
# Sign convention: below_m < 0 means the follower is ABOVE the boom tip.
# These are the SINGLE source of truth for both the live env call site
# (env_formation_sitl, once per step post sub-tick loop, perception geometry)
# and the offline replay-buffer recompute (tools/recompute_buffer_rewards.py,
# below_m decoded from stored obs[2]). Statelessness is a hard requirement:
# any per-episode memory here would break offline recomputability.

def above_boom_penalty(below_m: float) -> float:
    """Linear asymmetric penalty on any above-boom offset; exactly 0 at or
    below the boom, so the normal below-boom approach is untouched."""
    return -config.W_ABOVE_ASYM * max(0.0, -below_m)


def endpoint_ramp_penalty(below_m: float) -> float:
    """E5 (S20SPLIT): RAMPED endpoint penalty, armed W_ENDPOINT_RAMP_START_M
    above the boom tip. Exactly 0 below the arming point, so ordinary
    station-keeping is untaxed -- that is the whole point of the ramp, and
    what makes it 7x more selective than scaling the unramped term.

    ENDPOINT, not potential: this MOVES the optimum, unlike ceiling_potential
    below, which is policy-invariant PBRS and provably cannot.

    Stateless and a pure function of the stored endpoint, exactly like its
    siblings, so the live env call site and the offline replay-buffer
    recompute stay the same function of the same value.
    """
    up = -below_m
    return -config.W_ENDPOINT_RAMP * max(
        0.0, up - config.W_ENDPOINT_RAMP_START_M)


def ceiling_potential(below_m: float) -> float:
    """Potential Phi(s) <= 0: quadratic ramp arming CEILING_RAMP_START_M above
    the boom and steepening toward the 2.0 m bar (|COLLISION_ABOVE_BOOM_M|,
    the H1 bar). Used only through the PBRS difference below."""
    return -config.W_CEILING * max(0.0, (-below_m) - config.CEILING_RAMP_START_M) ** 2


def ceilrate_penalty(prev_below_m: float, below_m: float) -> float:
    """E5R Clause 1 (SIGNED 2026-08-19): RATE-aware ceiling penalty.

    Charges only when the follower is BOTH above the arming height AND
    climbing. E5's ramp taxes height and is blind to rate: at 1.9 m above the
    tip it charges the same whether the aircraft is parked or climbing at
    1 m/s. The measured failure is momentum -- all three D2a attempt-2 busts
    are arrest-lag events, and clean legs' median climb rate in the approach
    band is 0.000 m/s on every leg -- so a rate term is the selective one.

    Descending is never charged (max(0, .)); below the arm nothing is charged.
    ENDPOINT, not potential: it MOVES the optimum, which is the intent.

    Stateless pure function of the stored endpoint PAIR, exactly like its
    siblings, so the live env call site and the offline replay-buffer
    recompute are the same function of the same values.
    """
    up = -below_m
    if up <= config.CEILRATE_START_M:
        return 0.0
    up_rate = (up - (-prev_below_m)) / config.CEILRATE_DT_S
    return -config.W_CEILRATE * max(0.0, up_rate)


def ceiling_shaping_reward(prev_below_m: float, below_m: float) -> float:
    """Per-step shaping delta: asym(s') + gamma*Phi(s') - Phi(s).
    The PBRS pair telescopes over an episode (policy-ordering preserving);
    the asym term is a plain penalty. Applied at terminal steps with
    Phi(terminal) as-is (not zeroed) so the live term and the buffer
    recompute are the same function of the same stored endpoints.

    E5R Clause 1 joins here rather than as a separate term, following E5's own
    precedent (tools/reprice_terms.py: "ceiling_shaping_reward now CONTAINS
    the ramp"), so the existing "ceiling" reprice path prices it whole with no
    new decode assumptions."""
    return (above_boom_penalty(below_m)
            + endpoint_ramp_penalty(below_m)          # E5 / S20SPLIT
            + ceilrate_penalty(prev_below_m, below_m)  # E5R Clause 1
            + config.CEILING_SHAPE_GAMMA * ceiling_potential(below_m)
            - ceiling_potential(prev_below_m))


# --- OVSHAPE-20260816: endpoint no-pass shaping terms (module-level,
# STATELESS) — the signed longitudinal mirror of the SHAPE-585 pair above.
# Sign convention: behind_m < 0 means the follower is IN FRONT of the boom
# tip (past the no-pass line). Single source of truth for the live env call
# site AND the offline replay-buffer recompute (behind_m decoded from stored
# obs[0]); statelessness is the same hard requirement as above.

def precontact_reward(behind_error_m: float, cross_error_m: float,
                      below_error_m: float, in_contact_m3b: bool) -> float:
    """AMENDMENT R1 precontact tier-4 station-keeping pay, UNWEIGHTED.

    Stateless and pure, so the offline buffer reprice
    (tools/recompute_buffer_rewards.py --term precontact) can call THIS
    function on decoded observations and be bit-exact with the live call site
    in compute(). Extracted verbatim by R1ARM-723; the arithmetic is unchanged
    and pinned byte-for-byte by test_reward_precontact.py.

    Frame: the error axes SWITCH anchor with FSM state (_get_phase_targets:
    10 m standoff in HOLD_PRECONTACT and earlier, boom tip from DOCKING_CREEP
    on) -- so this band FOLLOWS the commanded target. Band (R1b, SIGNED
    2026-08-19): n is normalized by config.R_PRECONTACT_TIER's tolerances,
    elected TIER 1 (2.0, 2.0, 2.5), so n <= 1.0 spans tiers 1-4.

    ** THE FRAME IS NOT FIXED, and the rail DOES fire. ** An earlier version of
    this docstring called `in_contact_m3b` a structural rail that could never
    be active, arguing the bands are geometrically disjoint. That argument
    assumed `behind_error_m` is always measured from the 10 m standoff. It is
    NOT: `_get_phase_targets` returns the standoff only in HOLD_PRECONTACT and
    earlier, and returns the BOOM TIP (0.0, 0.45) from DOCKING_CREEP on. In the
    contact frame `behind_error_m == behind_m`, so this band very nearly
    COINCIDES with contact M3b and the rail suppresses it there.

    MEASURED on the carried s19-final buffer (R1ARM-723): 66.4% of transitions
    are in the CONTACT frame and 10.0% in the PRECONTACT one; of 13,596 in-band
    contact-frame rows, 10,825 are suppressed by this rail. Net, the term pays
    on 1.49% of transitions. That is the term behaving as specified -- it must
    not compete with r_contact_dwell inside contact M3b -- but it means the
    dense gradient R1 buys lands mostly in the PRECONTACT phase, not the
    contact one. Stated here so no reader re-derives the disjointness claim.
    """
    b_tol, c_tol, l_tol = config.MILESTONE_THRESHOLDS[config.R_PRECONTACT_TIER]
    n = max(abs(behind_error_m) / b_tol,
            abs(cross_error_m) / c_tol,
            abs(below_error_m) / l_tol)
    if in_contact_m3b or n > 1.0:
        return 0.0
    return min(1.0 - n, config.R_PRECONTACT_CAP)


def in_front_penalty(behind_m: float) -> float:
    """Linear asymmetric penalty on any in-front offset; exactly 0 at or
    behind the boom tip, so the normal approach from behind is untouched."""
    return -config.W_INFRONT_ASYM * max(0.0, -behind_m)


def nopass_potential(behind_m: float) -> float:
    """Potential Phi(s) <= 0: quadratic ramp arming NOPASS_RAMP_START_M in
    front of the tip and steepening toward the 2.0 m no-pass bar
    (|COLLISION_BEHIND_MIN|), SATURATED at NOPASS_CAP_M. The cap is the one
    signed departure from the ceiling form: the longitudinal excursion is
    unbounded, so an uncapped quadratic would dwarf the stored reward exactly
    when the policy is already failing; flat beyond the bar, the deep tail is
    carried by the linear asym term alone. Used only through the PBRS
    difference below."""
    d = max(0.0, (-behind_m) - config.NOPASS_RAMP_START_M)
    return -config.W_NOPASS * min(d, config.NOPASS_CAP_M) ** 2


def nopass_shaping_reward(prev_behind_m: float, behind_m: float) -> float:
    """Per-step shaping delta: asym(s') + gamma*Phi(s') - Phi(s).
    The PBRS pair telescopes over an episode (policy-ordering preserving);
    the asym term is a plain penalty. Terminal Phi as-is (not zeroed) so the
    live term and the buffer recompute are the same function of the same
    stored endpoints."""
    return (in_front_penalty(behind_m)
            + config.NOPASS_SHAPE_GAMMA * nopass_potential(behind_m)
            - nopass_potential(prev_behind_m))
