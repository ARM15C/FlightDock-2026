"""Curriculum stage manager with domain randomisation.

Tracks rollout count and stage progression (0-3). Each stage has
a fixed max rollouts with early-exit if M3b rate threshold is met.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Trajectory schedule (Phase E Batch 4 D6; supersedes Batch 3a
# TRAJECTORY_MIX_RATIOS + TRAJECTORY_CIRCLE_OMEGA_DEG_S).
# ---------------------------------------------------------------------------
# Per-stage dict colocating trajectory mix (racetrack/straight/circle
# probabilities) and stage-calibrated circle yaw rate. Canonical declaration
# per docs/C4_CAMPAIGN_PLAN_V2.md §3. Radii at TAS 24.7 m/s:
# S2 omega=4.0 deg/s -> radius ~354 m (11.5% margin in 400 m envelope);
# S3 omega=5.0 deg/s -> radius ~283 m (5.7% margin in 300 m envelope).
# Invariant: a stage may sample circle iff circle_omega_deg_s is not None.
# S0/S1 pattern: circle weight 0.00 + omega None (deferred per Alt-B).
# S4/S5 follow the same deferred pattern today: circle weight 0.00 +
# omega None, to make the schedule self-consistent at rest. Future
# activation at S4/S5 requires updating BOTH the circle mix weight AND
# circle_omega_deg_s in the same commit (the sample_domain_randomization
# ValueError guard enforces this atomicity if either is forgotten).
# config.CURRICULUM_STAGES supports stages 0-5 (per Batch 5). S4/S5
# circle remains deferred — racetrack/straight only — pending a
# separate calibrated batch with safety-envelope review.
TRAJECTORY_SCHEDULE: Dict[int, Dict[str, Any]] = {
    0: {"mix": {"racetrack": 1.00, "straight": 0.00, "circle": 0.00},
        "circle_omega_deg_s": None},
    1: {"mix": {"racetrack": 0.80, "straight": 0.20, "circle": 0.00},
        "circle_omega_deg_s": None},
    2: {"mix": {"racetrack": 0.70, "straight": 0.20, "circle": 0.10},
        "circle_omega_deg_s": 4.0},
    3: {"mix": {"racetrack": 0.70, "straight": 0.20, "circle": 0.10},
        "circle_omega_deg_s": 5.0},
    4: {"mix": {"racetrack": 0.80, "straight": 0.20, "circle": 0.00},
        "circle_omega_deg_s": None},  # circle deferred (mirrors S1);
                                       # activation requires paired
                                       # weight + omega update
    5: {"mix": {"racetrack": 0.80, "straight": 0.20, "circle": 0.00},
        "circle_omega_deg_s": None},  # circle deferred (mirrors S1);
                                       # activation requires paired
                                       # weight + omega update
}


# D2 (FD_CURRICULUM_V2): 11-sub-stage schedule, S0a/S0b 100% straight ramping to
# S5a 80% racetrack (inverts the v1 racetrack-dominant Stage 0). Circle deferred
# everywhere (weight 0 + omega None) — preserves the atomicity invariant in
# sample_domain_randomization. ADDITIVE: rebound onto TRAJECTORY_SCHEDULE only
# when the flag is ON; OFF => v1 schedule verbatim (byte-identical).
TRAJECTORY_SCHEDULE_V2: Dict[int, Dict[str, Any]] = {
    0:  {"mix": {"racetrack": 0.00, "straight": 1.00, "circle": 0.00}, "circle_omega_deg_s": None},
    1:  {"mix": {"racetrack": 0.00, "straight": 1.00, "circle": 0.00}, "circle_omega_deg_s": None},
    2:  {"mix": {"racetrack": 0.20, "straight": 0.80, "circle": 0.00}, "circle_omega_deg_s": None},
    3:  {"mix": {"racetrack": 0.20, "straight": 0.80, "circle": 0.00}, "circle_omega_deg_s": None},
    4:  {"mix": {"racetrack": 0.30, "straight": 0.70, "circle": 0.00}, "circle_omega_deg_s": None},
    5:  {"mix": {"racetrack": 0.40, "straight": 0.60, "circle": 0.00}, "circle_omega_deg_s": None},
    6:  {"mix": {"racetrack": 0.40, "straight": 0.60, "circle": 0.00}, "circle_omega_deg_s": None},
    7:  {"mix": {"racetrack": 0.50, "straight": 0.50, "circle": 0.00}, "circle_omega_deg_s": None},
    8:  {"mix": {"racetrack": 0.60, "straight": 0.40, "circle": 0.00}, "circle_omega_deg_s": None},
    9:  {"mix": {"racetrack": 0.70, "straight": 0.30, "circle": 0.00}, "circle_omega_deg_s": None},
    10: {"mix": {"racetrack": 0.80, "straight": 0.20, "circle": 0.00}, "circle_omega_deg_s": None},
}
# P1-b (FD_CURRICULUM_V3): 50/50 straight/racetrack EVERY stage (v3 design Q1);
# circle deferred everywhere (weight 0 + omega None — atomicity invariant kept).
TRAJECTORY_SCHEDULE_V3: Dict[int, Dict[str, Any]] = {
    i: {"mix": {"racetrack": 0.50, "straight": 0.50, "circle": 0.00},
        "circle_omega_deg_s": None}
    # RERAMP-811: range over MAX (26), not C1 (22) — appended stages fly the
    # same 50/50 straight/racetrack mix as every other v3 stage.
    for i in range(config.MAX_STAGE_INDEX_V3 + 1)
}

# NOTE: config sets FD_CURRICULUM_V2=True (module attr) under V3 to activate the
# env-side GPS ladder, so the V3 branch MUST be checked first here.
if config.FD_CURRICULUM_V3:
    TRAJECTORY_SCHEDULE = dict(TRAJECTORY_SCHEDULE_V3)
elif config.FD_CURRICULUM_V2:
    TRAJECTORY_SCHEDULE = dict(TRAJECTORY_SCHEDULE_V2)


def _v3_rehearsal_pool(cur: int) -> list:
    """RERAMP-811-J: the mastered rehearsal pool for active stage `cur` -- every
    LOWER index whose max sigma the holdout wall admits (<= the active stage's,
    same 1e-9 tolerance). Reads the SAME table _sample_v3 reads. Pure: no I/O,
    no truth field, no metric. Past C1 the ladder is appended, not sorted, so
    "lower index" no longer means "no rougher" -- this is the filter that
    assumption used to stand in for."""
    stages = config.CURRICULUM_STAGES
    act = stages[cur].wind_sigma_ms
    # A2A-20260901: an EMPTY sigma tuple means ZERO turbulence, so the ceiling
    # is 0.0 -- it is NOT "no ceiling". The old `None` sentinel disabled the
    # filter entirely, which was harmless only because index 0 was the sole
    # sigma-less stage and `range(1, 0)` is empty. The A2A ladder adds sigma-0
    # rungs ABOVE 30 lower indices, and under the old reading a still-air rung
    # would have rehearsed every rougher cell on the ladder -- including a
    # holdout sigma. Caught by
    # test_a_planted_holdout_sigma_leaves_the_pool_at_every_index, which plants
    # a 0.50 cell and asserts it never appears at ANY active index.
    act_max = max(act) if act else 0.0
    pool = []
    for i in range(1, cur):
        s = stages[i].wind_sigma_ms
        if not s or max(s) <= act_max + 1e-9:
            pool.append(i)
    return pool


class CurriculumManager:
    """Manages 6-stage curriculum for progressive DRL training.

    Parameters
    ----------
    initial_stage: int
        Starting stage (0-5).
    """

    def __init__(self, initial_stage: int = 0) -> None:
        self._stage = initial_stage
        self._rollout_count = 0
        self._stage_rollout_count = 0
        self._total_episodes = 0
        self._m3b_successes = 0
        # D1 (FD_DR_SEED_V2): private per-episode Generator set by the env before
        # each sample_domain_randomization() call. None => legacy global-RNG path.
        self._episode_rng = None
        # D2 competence floor: True when a timestep budget was reached but the
        # competence floor was NOT met (advance blocked). Telemetry-only.
        self._held_on_competence = False
        # P1-b (FD_CURRICULUM_V3) sampler state — inert on the v1/v2 paths.
        self._v3_c1_cycle: list = []            # shuffled stratified 32-cell cycle (C1)
        self._v3_c1_pos: int = 0
        self._v3_c1_counts: Dict[str, int] = {}  # realized per-cell episode counts (RC2)
        self._v3_last_cell: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def rollout_count(self) -> int:
        return self._rollout_count

    @property
    def held_on_competence(self) -> bool:
        """D2: True iff the last check_advance hit a stage budget but HELD because
        the competence floor was not met (blind-advance prevented)."""
        return self._held_on_competence

    # ------------------------------------------------------------------
    # Stage progression
    # ------------------------------------------------------------------

    def check_advance(
        self,
        rollout_count: int,
        mean_reward: float = 0.0,
        m3b_rate: float = 0.0,
        total_timesteps: int = 0,
        competence_ok: bool = True,
    ) -> bool:
        """Check if the curriculum should advance to the next stage.

        Called at rollout boundaries (every n_steps * n_envs transitions).

        Returns True if stage was advanced.
        """
        self._rollout_count = rollout_count
        self._stage_rollout_count += 1

        # P1-b (FD_CURRICULUM_V3): sealed offline gates are the SOLE promote
        # authority — in-loop early-exit and budget auto-advance are DEAD here.
        # Advance happens only via promote() (gate-runner PROMOTE verdict).
        if config.FD_CURRICULUM_V3:
            return False

        if self._stage >= max(config.CURRICULUM_STAGES):
            return False  # Already at final stage

        stage_cfg = config.CURRICULUM_STAGES[self._stage]

        # Early exit: advance if M3b rate above threshold AND min rollouts met
        if (
            self._stage_rollout_count >= config.EARLY_EXIT_MIN_ROLLOUTS
            and m3b_rate >= stage_cfg.m3b_early_exit_rate
        ):
            self._advance()
            return True

        # D2 (FD_CURRICULUM_V2): timestep-budget advance supersedes rollout count.
        # config.CUMULATIVE_TIMESTEPS is None in v1 => legacy max_rollouts path.
        # NOTE: the operator-proposed competence floor (simultaneous f9>0 in >=3
        # of the last 5 eval episodes) is a callbacks-level gate PENDING operator
        # approval (the D3-authorization gate); it is intentionally NOT wired
        # here — this implements the timestep budget + the m3b early-exit only.
        if config.CUMULATIVE_TIMESTEPS is not None:
            if total_timesteps >= config.CUMULATIVE_TIMESTEPS[self._stage]:
                # Competence floor: advance only if the
                # agent shows competence at this stage; else HOLD + flag — never
                # advance blind into a harder rung. competence_ok is computed by
                # the callback from a >=3-of-last-5 window of an EXISTING online
                # per-episode signal (perception-side M3b dwell > 0); it is NOT a
                # new env metric and NOT the truth-dependent offline grader f9.
                if competence_ok:
                    self._held_on_competence = False
                    self._advance()
                    return True
                self._held_on_competence = True
                return False
            return False

        # Max rollouts: advance unconditionally (v1)
        if self._stage_rollout_count >= stage_cfg.max_rollouts:
            self._advance()
            return True

        return False

    def _advance(self) -> None:
        """Move to the next stage.

        RERAMP-811: under V3 the ceiling is PROMOTION_CEILING_V3 (C1), NOT the
        ladder length — the appended re-ramp cells 23..26 are easier than C1
        and must never be reached by auto-advance (see the config note).
        """
        ceiling = max(config.CURRICULUM_STAGES)
        if config.FD_CURRICULUM_V3:
            ceiling = min(ceiling, config.PROMOTION_CEILING_V3)
        self._stage = min(self._stage + 1, ceiling)
        self._stage_rollout_count = 0

    # ------------------------------------------------------------------
    # P1-b (FD_CURRICULUM_V3): gate-driven progression + A5 sampler cursor
    # ------------------------------------------------------------------

    def promote(self) -> int:
        """V3: sealed-gate PROMOTE verdict -> advance one rung. The gate-runner
        verdict is the SOLE advance authority under FD_CURRICULUM_V3
        (check_advance is inert there)."""
        self._advance()
        return self._stage

    def demote(self) -> int:
        """V3 Q5: confirmed-strikes demotion -> back one rung (checkpoint
        reload + replay-buffer keep are the caller's responsibility)."""
        self._stage = max(self._stage - 1, 0)
        self._stage_rollout_count = 0
        return self._stage

    @property
    def stage_label(self) -> str:
        if config.FD_CURRICULUM_V3:
            return config.STAGE_LABELS_V3.get(self._stage, str(self._stage))
        return str(self._stage)

    def get_v3_sampler_state(self) -> Dict[str, Any]:
        """A5 snapshot: sampler cursor (stage + C1 cycle order/position + realized
        per-cell counts)."""
        return {"stage": self._stage,
                "stage_rollout_count": self._stage_rollout_count,
                "c1_cycle": [list(c) for c in self._v3_c1_cycle],
                "c1_pos": self._v3_c1_pos,
                "c1_counts": dict(self._v3_c1_counts)}

    def set_v3_sampler_state(self, state: Dict[str, Any]) -> None:
        self._stage = int(state["stage"])
        self._stage_rollout_count = int(state.get("stage_rollout_count", 0))
        self._v3_c1_cycle = [tuple(c) for c in state.get("c1_cycle", [])]
        self._v3_c1_pos = int(state.get("c1_pos", 0))
        self._v3_c1_counts = dict(state.get("c1_counts", {}))

    # ------------------------------------------------------------------
    # Domain randomisation
    # ------------------------------------------------------------------

    def sample_domain_randomization(self) -> Dict[str, Any]:
        """Sample domain randomisation parameters for current stage.

        Returns a dict that can be passed to env.reset(options=...).
        """
        # P1-b (FD_CURRICULUM_V3): dedicated deterministic cell sampler.
        if config.FD_CURRICULUM_V3:
            return self._sample_v3()

        stage_cfg = config.CURRICULUM_STAGES[self._stage]

        # D1 (FD_DR_SEED_V2): if the env supplied a private per-episode Generator,
        # ALL draws use it (rng.*) — ZERO global random.*/np.random.* on this path
        # (SB3 RNG-isolation, Correction 1). rng is None => legacy global-`random`
        # path runs verbatim (byte-identical).
        rng = self._episode_rng
        if rng is not None:
            wind_speed = rng.uniform(0.0, stage_cfg.wind_max_ms)
            wind_dir = rng.uniform(0.0, 360.0)
        else:
            wind_speed = random.uniform(0.0, stage_cfg.wind_max_ms)
            wind_dir = random.uniform(0.0, 360.0)
        wind_turb = stage_cfg.wind_turb

        # Phase E Batch 4 D6: sample leader trajectory profile per Alternative B
        # mix + stage-calibrated circle yaw rate.
        stage_schedule = TRAJECTORY_SCHEDULE.get(
            self._stage, TRAJECTORY_SCHEDULE[max(TRAJECTORY_SCHEDULE)]  # clamp beyond table
        )
        mix = stage_schedule["mix"]
        if rng is not None:
            _keys = list(mix.keys())
            _w = np.asarray(list(mix.values()), dtype=float)
            _w = _w / _w.sum()
            leader_profile = str(rng.choice(_keys, p=_w))
        else:
            leader_profile = random.choices(
                population=list(mix.keys()),
                weights=list(mix.values()),
                k=1,
            )[0]
        leader_omega = None
        if leader_profile == "circle":
            leader_omega = stage_schedule["circle_omega_deg_s"]
            if leader_omega is None:
                # Invariant: a stage whose circle_omega_deg_s is None must have
                # circle mix weight 0. If we reach here, TRAJECTORY_SCHEDULE has
                # an inconsistent entry for this stage (e.g. S4/S5 with circle
                # weight > 0 while omega is still a TBD placeholder).
                raise ValueError(
                    f"Stage {self._stage}: circle profile sampled but "
                    f"circle_omega_deg_s is None. TRAJECTORY_SCHEDULE "
                    f"inconsistency — stages with circle_omega_deg_s=None "
                    f"must have circle mix weight=0."
                )

        return {
            "wind_speed_ms": wind_speed,
            "wind_dir_deg": wind_dir,
            "wind_turb": wind_turb,
            "gps_accuracy": stage_cfg.gps_accuracy,
            # D2 (FD_CURRICULUM_V2): GPS-ladder inputs. Inert in v1 — dropout
            # defaults False/0 and the env applies the ladder only when v2.
            "gps_cep_m": stage_cfg.gps_accuracy,
            "gps_dropout_enabled": stage_cfg.gps_dropout_enabled,
            "gps_dropout_max_per_episode": stage_cfg.gps_dropout_max_per_episode,
            "warmup_target": stage_cfg.warmup_target,
            "action_scale": stage_cfg.action_scale,
            "leader_profile": leader_profile,
            "leader_omega_deg_s": leader_omega,
        }

    def _sample_v3(self) -> Dict[str, Any]:
        """P1-b v3 cell sampler: fixed-W + FROM-180 (never drawn), 50/50
        trajectory, 70/20/10 rehearsal from W2 (current/uniform-mastered/S0),
        C1 stratified 32-cell cycle. Deterministic by construction: every draw
        uses the per-episode rng (refuses the global-random path).

        Side effect: writes the FD_FDM_WIND_* env vars (worker-process-local)
        that env._launch_fdm() consumes later in the SAME reset — the #465
        channel; env_formation_sitl.py is untouched. Evals set
        FD_V3_REHEARSAL_DISABLE=1 so sealed gates fly current-cell only.
        """
        rng = self._episode_rng
        if rng is None:
            raise RuntimeError(
                "FD_CURRICULUM_V3 requires the per-episode RNG (set "
                "FD_DR_SEED_V2=1 and seed the env) — refusing the "
                "nondeterministic global-random path.")
        cur = self._stage
        rehearsal_off = os.environ.get("FD_V3_REHEARSAL_DISABLE", "0") == "1"
        if cur == config.C1_STAGE_INDEX_V3:
            cell_stage, leader_profile = self._draw_c1_cell(rng)
        else:
            cell_stage = cur
            if not rehearsal_off and cur >= config.V3_REHEARSAL_START_INDEX:
                u = float(rng.random())
                if u < 0.10:
                    cell_stage = 0                          # S0 rehearsal slot
                elif u < 0.30:
                    # RERAMP-811-J (2026-08-24): uniform over the mastered rungs the
                    # holdout wall below ADMITS, not over the bare index range. This
                    # read `rng.integers(1, cur)` -- index == difficulty -- and the
                    # ladder append put sigma-0.15 P1a at index 30 above sigma-0.25/
                    # 0.30 rungs, so 15 of 29 draws tripped the wall; leg 1 died at
                    # reset 26 after passing every static launch witness. When nothing
                    # is filtered, pool[integers(0, n)] is bit-identical to the old
                    # call (pinned in test_rehearsal_pool_sigma_order.py). An empty
                    # pool draws anyway (same rng consumption) and flies the current
                    # cell; it never raises and never falls through to S0.
                    pool = _v3_rehearsal_pool(cur)
                    k = int(rng.integers(0, max(len(pool), 1)))
                    cell_stage = pool[k] if pool else cur
            leader_profile = ("straight" if float(rng.random()) < 0.5
                              else "racetrack")
        cfg = config.CURRICULUM_STAGES[cell_stage]
        # A5 diagonal sigma semantics:
        #  - C1 (E1=a): the drawn cell supplies WIND only; sigma/L come from
        #    C1's own stage cfg (fixed 0.30) — never the cell's historical
        #    sigma.
        #  - W-stage rehearsal: the mastered cell flies at its OWN sigma
        #    (retention semantics), which the diagonal makes <= current by
        #    construction.
        sig_cfg = (config.CURRICULUM_STAGES[cur]
                   if cur == config.C1_STAGE_INDEX_V3 else cfg)
        # A5 (e): the holdout wall IN CODE — on the default path every
        # training draw's sigma must be a curriculum value and never exceed
        # the active stage's. A set FD_V3_SIGMA_OVERRIDE_MS bypasses this
        # wall BY DESIGN (probe/eval semantics; guarded by the env_contract
        # ack), so the wall asserts only when the table is the source.
        if not getattr(config, "V3_SIGMA_IS_OVERRIDE", False):
            if sig_cfg.wind_sigma_ms:
                # ONE FACT, ONE NAME (config.V3_CURRICULUM_SIGMAS). This tuple
                # used to be an inline literal (0.10, 0.20, 0.30) and went
                # stale when D1a (sigma 0.15) was inserted at index 14: the
                # tests learned the new rung, the runtime did not, and W1418
                # died on its first env.reset() after passing all fifteen
                # launch gates.
                assert all(round(s, 2) in config.V3_CURRICULUM_SIGMAS
                           for s in sig_cfg.wind_sigma_ms), (
                    f"holdout wall: non-curriculum sigma "
                    f"{sig_cfg.wind_sigma_ms} at cell {cell_stage}")
                # The sealed H3 values, named EXPLICITLY. Previously they were
                # caught only by their absence from the permitted list, so the
                # wall never mentioned the very values it exists to seal — and
                # any widening of that list would have silently unsealed them.
                assert not any(round(s, 2) in config.V3_HOLDOUT_SIGMAS
                               for s in sig_cfg.wind_sigma_ms), (
                    f"holdout wall: H3 HOLDOUT sigma "
                    f"{sig_cfg.wind_sigma_ms} at cell {cell_stage}")
                _cur_sig = config.CURRICULUM_STAGES[cur].wind_sigma_ms
                assert (not _cur_sig or
                        max(sig_cfg.wind_sigma_ms) <= max(_cur_sig) + 1e-9), (
                    f"holdout wall: rehearsal sigma {sig_cfg.wind_sigma_ms} "
                    f"exceeds active stage {cur} sigma {_cur_sig}")
        wind = float(cfg.wind_max_ms)                       # fixed-W: EXACT wind
        # --- #465 FDM Dryden channel (read by _launch_fdm this reset) ---
        os.environ["FD_FDM_WIND_SPEED_MS"] = f"{wind:.3f}"
        os.environ["FD_FDM_WIND_FROM_DEG"] = f"{float(cfg.wind_from_deg):.1f}"
        if sig_cfg.wind_sigma_ms:
            os.environ["FD_FDM_WIND_SIGMA_MS"] = ",".join(
                f"{s:.3f}" for s in sig_cfg.wind_sigma_ms)
            os.environ["FD_FDM_WIND_LENGTH_M"] = ",".join(
                f"{l:.1f}" for l in sig_cfg.wind_L_m)
        else:
            os.environ.pop("FD_FDM_WIND_SIGMA_MS", None)
            os.environ.pop("FD_FDM_WIND_LENGTH_M", None)
        cell = {
            "wind_speed_ms": wind,
            "wind_dir_deg": float(cfg.wind_from_deg),
            # SIM_WIND_TURB knob is inert on json; the real turbulence is the
            # FDM sigma written above (and echoed in wind_sigma_ms below).
            "wind_turb": cfg.wind_turb,
            "gps_accuracy": cfg.gps_accuracy,
            "gps_cep_m": cfg.gps_accuracy,
            "gps_dropout_enabled": False,
            "gps_dropout_max_per_episode": 0,
            "warmup_target": cfg.warmup_target,
            "action_scale": cfg.action_scale,
            "leader_profile": leader_profile,
            "leader_omega_deg_s": None,
            # v3 telemetry (additive metrics columns)
            "v3_cell_stage": cell_stage,
            "v3_cell_label": config.STAGE_LABELS_V3[cell_stage],
            "v3_rehearsal": bool(cell_stage != cur),
            "wind_sigma_ms": list(sig_cfg.wind_sigma_ms),
            "wind_L_m": list(sig_cfg.wind_L_m),
        }
        self._v3_last_cell = cell
        return cell

    def _draw_c1_cell(self, rng) -> tuple:
        """C1 (RC2; grid extended by T-191/E4): shuffled stratified cycle — a
        full permutation of the 40 mastered cells (W1..W20 x
        straight/racetrack) per cycle, reshuffled at wrap; realized per-cell
        counts logged (gate eligibility requires >=1 complete episode in
        every cell)."""
        if not self._v3_c1_cycle or self._v3_c1_pos >= len(self._v3_c1_cycle):
            # A5 diagonal: D1/D2 are EXCLUDED — their winds duplicate W12/W16
            # and the SIGNED C1 grid (E1=a) is wind x trajectory = W1..W20 x 2
            # = 40 cells, flown at C1's own FIXED sigma 0.30 (the sampler
            # overrides the drawn cell's historical sigma — see _sample_v3).
            cells = [(s, t) for s in range(2, config.C1_STAGE_INDEX_V3)
                     if not config.STAGE_LABELS_V3[s].startswith("D")
                     for t in ("straight", "racetrack")]
            order = rng.permutation(len(cells))
            self._v3_c1_cycle = [cells[int(k)] for k in order]
            self._v3_c1_pos = 0
        cell_stage, traj = self._v3_c1_cycle[self._v3_c1_pos]
        self._v3_c1_pos += 1
        # R1.7: per-cell coverage key carries the sigma axis explicitly
        # (V3 table addressed directly — the CURRICULUM_STAGES alias only
        # exists under FD_CURRICULUM_V3=1 and this method is also exercised
        # standalone by the detector tests)
        sig = config.CURRICULUM_STAGES_V3[config.C1_STAGE_INDEX_V3].wind_sigma_ms
        key = (f"{config.STAGE_LABELS_V3[cell_stage]}:{traj}"
               f":s{max(sig):.2f}" if sig else
               f"{config.STAGE_LABELS_V3[cell_stage]}:{traj}")
        self._v3_c1_counts[key] = self._v3_c1_counts.get(key, 0) + 1
        return int(cell_stage), traj

    def get_reward_weights(self) -> Dict[str, float]:
        """Get stage-dependent reward weights."""
        w = config.STAGE_WEIGHTS[self._stage]
        return {
            "w_ref": w.w_ref,
            "w_prox": w.w_prox,
            "w_dwell": w.w_dwell,
            "w_safe": w.w_safe,
            "w_smooth": w.w_smooth,
            "w_wake": w.w_wake,
        }

    def get_stage_params(self) -> Dict[str, Any]:
        """Get full stage configuration for logging."""
        stage_cfg = config.CURRICULUM_STAGES[self._stage]
        return {
            "stage": self._stage,
            "max_rollouts": stage_cfg.max_rollouts,
            "wind_max_ms": stage_cfg.wind_max_ms,
            "wind_turb": stage_cfg.wind_turb,
            "gps_accuracy": stage_cfg.gps_accuracy,
            "stage_rollout_count": self._stage_rollout_count,
        }
