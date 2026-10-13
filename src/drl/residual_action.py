"""Residual action processor — scale, clamp, rate-limit, and store applied residual.

ACTION_DIM=3 (2026-05-06 cross-axis restoration): the DRL agent outputs
raw actions in [-1, 1]^3 — a 3-element vector [speed, heading, altitude].
This module:

  1. Scales by ACTION_SCALES → 3D residual [speed, heading, altitude]
  2. **C3X Intervention 2 — Behind-axis clamp at Site (ii):** clamps
     the speed (behind-axis) residual to
     ``[CLAMP_BEHIND_NEGATIVE_MS, CLAMP_BEHIND_POSITIVE_MS]``
     = ``[-1.0, +0.10] m/s`` after scaling and BEFORE integration with
     the baseline PID's speed setpoint. Path A(i) ULTRA_NEAR-anchored
     bracket midpoint per ``docs/C3X_PRECOMMIT_SYNTHESIS.md`` §2 and
     operator adjudication 2026-04-08. Negative authority is preserved
     at full natural range; positive authority bounded at +0.10 m/s.
  3. Heading residual clamp (ACTION_DIM=3 restoration, ±3 deg symmetric):
     clamps the heading delta to
     ``[CLAMP_HEADING_NEGATIVE_DEG, CLAMP_HEADING_POSITIVE_DEG]``
     before adding to baseline.heading_deg. Downstream baseline rate
     limiter (HEADING_RATE_*_DPS) governs achieved rate.
  4. Adds (clamped) speed delta to baseline.speed_ms
  5. Adds (clamped) heading delta to baseline.heading_deg (mod 360)
  6. Adds altitude delta to baseline.altitude_m
  7. Clamps speed to [SAFE_FLOOR_MS, SAFE_MAX_MS] (env safety bound)
  8. Clamps altitude to [leader ± ALT_OFFSET_MAX_M] (env safety bound)
  9. Computes the *applied* residual (final - baseline, post-clamp).
     The applied_residual array has shape (3,) with layout
     ``[speed_residual, heading_residual, altitude_residual]``,
     consumed by the obs builder's prev-action-memory channels at
     obs[35]/[36]/[37].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class BaselineCommand:
    """Baseline controller output for one tick."""
    speed_ms: float
    heading_deg: float
    altitude_m: float


@dataclass
class ProcessedAction:
    """Final command after residual application and safety clamping.

    ``applied_residual`` is shape (3,) with layout
    [speed_residual, heading_residual, altitude_residual]; consumed by
    the obs builder's prev-action-memory channels at obs[35]/[36]/[37].
    """
    speed_ms: float
    heading_deg: float
    altitude_m: float
    applied_residual: np.ndarray  # shape (3,) — [speed_res, heading_res, alt_res]


class ResidualActionProcessor:
    """Processes raw policy actions into safe flight commands.

    Maintains heading slew state for rate-limiting across ticks.
    """

    def __init__(self) -> None:
        self._prev_heading_cmd: float = 0.0
        # EMA/slew on policy speed delta (not baseline)
        self._ema_policy_speed: float = 0.0
        self._prev_policy_speed: float = 0.0

    def seed_heading(self, heading_deg: float) -> None:
        """Seed the slew state from the baseline heading at warmup handoff.

        Must be called before the first process() call each episode.
        """
        self._prev_heading_cmd = heading_deg % 360.0

    def reset(self) -> None:
        """Clear slew and filter state (call at episode start)."""
        self._prev_heading_cmd = 0.0
        self._ema_policy_speed = 0.0
        self._prev_policy_speed = 0.0

    def pin_zero(self) -> None:
        """Pin all filter state to zero. Called during force-zero bypass."""
        self._ema_policy_speed = 0.0
        self._prev_policy_speed = 0.0

    # Ramp-in period: linearly scale residual authority from 0→1 over this
    # many ticks after DRL handoff. Prevents discontinuity at warmup→DRL.
    RAMP_IN_TICKS: int = 40  # 2 seconds at 20Hz

    def process_reference(
        self,
        raw_action: np.ndarray,
        action_scale: float = 1.0,
        tick_count: int = 0,
    ) -> ProcessedAction:
        """Scale raw policy action into position offsets (RFL reference mode).

        ACTION_DIM=3: raw_action is shape (3,) — [behind_offset,
        cross_offset, below_offset] in normalised [-1, 1] units. The
        applied_residual is emitted as shape (3,) for downstream
        consumers indexing [0]/[1]/[2] (e.g. the env's
        ``set_position_offset`` call site).

        Returns ProcessedAction with applied_residual = position offsets
        [behind_m, cross_m, below_m] in meters. speed_ms/heading_deg/
        altitude_m are NaN sentinels — any downstream consumer reading
        these in reference mode will produce NaN propagation (fail loud).
        """
        raw = np.asarray(raw_action, dtype=np.float64)
        scales = np.array(config.ACTION_SCALES_REFERENCE, dtype=np.float64)

        # Ramp-in: linearly increase authority over first N ticks
        if tick_count < self.RAMP_IN_TICKS:
            ramp_factor = tick_count / self.RAMP_IN_TICKS
            raw = raw * ramp_factor

        scaled = raw * scales * action_scale  # shape (3,)

        applied_3d = np.array(
            [float(scaled[0]), float(scaled[1]), float(scaled[2])],
            dtype=np.float32,
        )

        return ProcessedAction(
            speed_ms=float('nan'),
            heading_deg=float('nan'),
            altitude_m=float('nan'),
            applied_residual=applied_3d,
        )

    def process(
        self,
        baseline: BaselineCommand,
        raw_action: np.ndarray,    # shape (3,) in [-1, 1] — [speed, heading, altitude]
        range_m: float,
        leader_alt_m: float,
        dt_s: float,
        action_scale: float = 1.0,
        tick_count: int = 0,
        crrl_bounds: np.ndarray | None = None,  # P3-CRRL clip bounds
        forward_bias: float = 0.0,  # Env-computed gated bias
        vert_offset_down_m: float = 0.0,  # Env-computed gated down-offset (VERTOFF-20260902)
    ) -> ProcessedAction:
        """Apply residual action to baseline with safety clamping.

        ACTION_DIM=3 (cross-axis restored 2026-05-06): raw_action is a
        3-element vector ``[speed, heading, altitude]`` in [-1, 1]. The
        behind-axis (speed) residual is clamped to
        ``[CLAMP_BEHIND_NEGATIVE_MS, CLAMP_BEHIND_POSITIVE_MS] = [-1.0,
        +0.10] m/s`` (asymmetric, ULTRA_NEAR-anchored) before EMA/slew.
        The heading residual is clamped symmetrically to
        ``[CLAMP_HEADING_NEGATIVE_DEG, CLAMP_HEADING_POSITIVE_DEG]
        = ±3 deg`` per tick before adding to baseline.heading_deg.

        Parameters
        ----------
        baseline: BaselineCommand
            Baseline controller output for this tick.
        raw_action: np.ndarray
            Raw policy output in [-1, 1]^3 — [speed, heading, altitude].
        range_m: float
            Current range to leader (retained for API compatibility).
        leader_alt_m: float
            Leader altitude MSL (for altitude clamp bounds).
        dt_s: float
            Tick delta-time in seconds.
        action_scale: float
            Curriculum-dependent action authority multiplier.
        tick_count: int
            Current DRL tick count (for ramp-in).
        crrl_bounds: np.ndarray or None
            Per-axis clip bounds for CRRL (Campaign 3 P3). Shape (3,) clips
            symmetrically to ``[-crrl_bounds, +crrl_bounds]``; shape (2, 3)
            is the B'+C per-sign form ``[neg, pos]`` clipping to
            ``[-neg, +pos]`` (era bprime-c-v1). Applied before per-axis
            safety processing.

        Returns
        -------
        ProcessedAction
            Final commands and the applied (post-clamp) residual.
            ``applied_residual`` has shape (3,) — layout
            [speed_residual, heading_residual, altitude_residual].
        """
        raw = np.asarray(raw_action, dtype=np.float64)
        scales = np.array(config.ACTION_SCALES, dtype=np.float64)

        # Ramp-in: linearly increase residual authority over first N ticks
        if tick_count < self.RAMP_IN_TICKS:
            ramp_factor = tick_count / self.RAMP_IN_TICKS
            raw = raw * ramp_factor

        scaled = raw * scales * action_scale  # shape (3,) — [speed, heading, altitude]

        # P3-CRRL: clip scaled residual before per-axis safety processing.
        # Legacy shape (3,) = symmetric bounds; shape (2, 3) = B'+C per-sign
        # [neg_bounds, pos_bounds] (bprime-c-v1, §5 SIGNED Option A) — clip to
        # [-neg, +pos] so elected floors apply only on their measured sign.
        if crrl_bounds is not None:
            b = np.asarray(crrl_bounds, dtype=np.float64)
            if b.ndim == 2:
                scaled = np.clip(scaled, -b[0], b[1])
            else:
                scaled = np.clip(scaled, -b, b)

        # --- Speed: policy delta extraction, C3X clamp, then EMA/slew/baseline addition ---
        # EMA/slew act on POLICY DELTA only, preserving MRRL identity contract.
        # C3X Intervention 2: clamp the raw scaled[0] (speed delta) to
        # [CLAMP_BEHIND_NEGATIVE_MS, CLAMP_BEHIND_POSITIVE_MS] BEFORE
        # the EMA/slew pipeline. This forces the actor's effective speed
        # delta to operate within the baseline's closure-cap absorption
        # envelope. The clamp is asymmetric: full natural negative
        # authority preserved (-1.0 m/s), positive authority bounded at
        # +0.10 m/s (Path A(i) ULTRA_NEAR-anchored midpoint).
        policy_delta = float(scaled[0])
        policy_delta = float(np.clip(
            policy_delta,
            config.CLAMP_BEHIND_NEGATIVE_MS,
            config.CLAMP_BEHIND_POSITIVE_MS,
        ))

        # V4: EMA on policy delta (alpha < 1.0 enables filtering)
        if config.EMA_ALPHA_SPEED < 1.0:
            policy_delta = (config.EMA_ALPHA_SPEED * policy_delta
                            + (1.0 - config.EMA_ALPHA_SPEED) * self._ema_policy_speed)
            self._ema_policy_speed = policy_delta

        # V4: Slew-rate on policy delta
        max_delta_change = config.SPEED_SLEW_RATE_MPS2 * dt_s
        delta_change = policy_delta - self._prev_policy_speed
        delta_change = float(np.clip(delta_change, -max_delta_change, max_delta_change))
        policy_delta = self._prev_policy_speed + delta_change
        self._prev_policy_speed = policy_delta

        # V3: Bias computed by env, passed as parameter
        # Final: baseline + bias + filtered (clamped) policy delta
        speed_raw = baseline.speed_ms + forward_bias + policy_delta
        speed_cmd = float(np.clip(speed_raw, config.SAFE_FLOOR_MS, config.SAFE_MAX_MS))

        # --- Heading: clamp residual delta then add to baseline heading ---
        # Symmetric ±CLAMP_HEADING_*_DEG bound. Downstream baseline
        # rate-limiter governs achieved-rate; this clamp constrains
        # per-tick residual amplitude.
        heading_delta_deg = float(np.clip(
            float(scaled[1]),
            config.CLAMP_HEADING_NEGATIVE_DEG,
            config.CLAMP_HEADING_POSITIVE_DEG,
        ))
        heading_cmd = (float(baseline.heading_deg) + heading_delta_deg) % 360.0
        self._prev_heading_cmd = heading_cmd

        # --- Altitude: baseline + residual, clamped to leader ± offset ---
        # VERTOFF-20260902: gated down-offset subtracted OUTSIDE the residual clip,
        # mirroring forward_bias on the speed axis (residual_action.py:244) — an
        # independent lever, not part of the policy's authority envelope.
        alt_raw = baseline.altitude_m + scaled[2] - vert_offset_down_m
        alt_floor = leader_alt_m - config.ALT_OFFSET_MAX_M
        alt_ceil = leader_alt_m + config.ALT_OFFSET_MAX_M
        alt_cmd = float(np.clip(alt_raw, alt_floor, alt_ceil))

        # #453: the #390 C1 inner residual-layer belly clamp was RETIRED here.
        # The authoritative belly clamp is the env command chokepoint
        # (env_formation_sitl._apply_belly_clamp, #452) which covers BOTH residual
        # modes; this actuator-only layer was reference-mode-blind and gate-
        # mismatched (range<5.0 vs the FSM-state gate). Single chokepoint now.

        # --- Applied residual (post-clamp) ---
        # Layout [speed_residual, heading_residual, altitude_residual]
        # consumed by the obs builder's prev-action-memory channels at
        # obs[35]/[36]/[37].
        applied = np.array([
            speed_cmd - baseline.speed_ms,
            heading_delta_deg,
            alt_cmd - baseline.altitude_m,
        ], dtype=np.float32)

        return ProcessedAction(
            speed_ms=speed_cmd,
            heading_deg=heading_cmd,
            altitude_m=alt_cmd,
            applied_residual=applied,
        )
