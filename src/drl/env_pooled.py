"""Pooled variant of FormationSITLEnv — uses pre-warmed SITL pool.

Instead of running a full 3-4 minute warmup on every reset(), this env
grabs a pre-warmed instance from the SITLPool. This reduces reset time
from ~240 seconds to near-zero (if a pool instance is ready).

The pool runs warmup in background threads, so new instances are being
prepared while the DRL agent steps through other environments.

Usage
-----
    from drl.sitl_pool import SITLPool
    from drl.env_pooled import PooledFormationSITLEnv

    pool = SITLPool(n_envs=3, speedup=2.0)
    pool.start()

    env = PooledFormationSITLEnv(env_id=0, pool=pool, speedup=2.0)
    obs, info = env.reset()
    # ... step loop ...
    env.close()
    pool.stop()
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from . import config
from .env_formation_sitl import FormationSITLEnv
from .observation_builder import FollowerState, LeaderState
from .sitl_pool import SITLPool, WarmupResult


class PooledFormationSITLEnv(FormationSITLEnv):
    """FormationSITLEnv variant that acquires pre-warmed SITL from pool.

    Overrides ``reset()`` to skip the warmup phase entirely, instead
    taking ownership of a pre-warmed instance from the shared pool.

    All other methods (step, close, etc.) are inherited from the parent.

    Parameters
    ----------
    env_id: int
        Instance ID for port isolation and pool slot assignment.
    pool: SITLPool
        Shared pool of pre-warmed SITL instances.
    speedup: float
        SITL speedup factor.
    kwargs:
        Additional arguments passed to FormationSITLEnv.
    """

    def __init__(
        self,
        env_id: int = 0,
        pool: Optional[SITLPool] = None,
        speedup: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(env_id=env_id, speedup=speedup, **kwargs)
        self._pool = pool
        self._acquired_result: Optional[WarmupResult] = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Acquire pre-warmed SITL from pool instead of running warmup.

        Falls back to full warmup (parent reset) if pool is unavailable.
        """
        super(FormationSITLEnv, self).reset(seed=seed)

        # D1 (FD_DR_SEED_V2): per-episode deterministic RNG for pooled DR draws,
        # mirroring the non-pooled path (pooled reset bypasses the parent reset).
        # Flag OFF => None => legacy global-RNG DR (byte-identical) and the
        # episode index is left untouched. NOTE: pool slot-acquisition order is
        # not per-episode deterministic (documented non-goal); strict-determinism
        # validation uses the non-pooled path.
        if self._dr_seed_v2 and self._initial_seed is not None:
            _ep_idx = self._episode_index
            self._episode_index += 1
            _ss = np.random.SeedSequence(
                [int(self._initial_seed), int(self.env_id), int(_ep_idx)]
            )
            self._episode_rng = np.random.default_rng(_ss.generate_state(4))
            self._episode_weather_seed = int(_ss.generate_state(8)[4])
        else:
            self._episode_rng = None
            self._episode_weather_seed = None
        if self.curriculum_manager is not None:
            self.curriculum_manager._episode_rng = self._episode_rng

        # Kill previous episode's processes
        self._kill_procs()

        # Domain randomisation
        wind_spd = self.wind_speed_ms
        wind_dir = 0.0
        wind_turb = self.wind_turb
        gps_accuracy = 0.3
        gps_dropout_enabled = False  # D2: GPS dropout windows (v2 S4a+ only)
        gps_dropout_max = 0
        warmup_target = "M2"
        action_scale = 1.0
        if self.curriculum_manager is not None:
            dr = self.curriculum_manager.sample_domain_randomization()
            wind_spd = dr.get("wind_speed_ms", wind_spd)
            wind_dir = dr.get("wind_dir_deg", wind_dir)
            wind_turb = dr.get("wind_turb", wind_turb)
            gps_accuracy = dr.get("gps_accuracy", gps_accuracy)
            gps_dropout_enabled = dr.get("gps_dropout_enabled", gps_dropout_enabled)
            gps_dropout_max = dr.get("gps_dropout_max_per_episode", gps_dropout_max)
            warmup_target = dr.get("warmup_target", "M2")
            action_scale = dr.get("action_scale", 1.0)
        self._action_scale = action_scale

        # Try to acquire from pool
        if self._pool is not None:
            try:
                t0 = time.monotonic()
                result = self._pool.acquire(self.env_id, timeout=300.0)
                acquire_time = time.monotonic() - t0

                # Take ownership of all handles
                self._adopt_warmup_result(result)

                tag = f"[DRL env_id={self.env_id}]"
                logger.info(f"{tag} Pool acquire: {acquire_time:.1f}s wait "
                            f"(warmup was {result.warmup_wall_seconds:.1f}s)")

            except TimeoutError:
                # Fallback: run warmup inline (as the parent would)
                tag = f"[DRL env_id={self.env_id}]"
                logger.warning(f"{tag} Pool timeout — falling back to inline warmup")
                self._launch_sitl()
                self._launch_leader(wind_spd, wind_dir, wind_turb,
                                    weather_seed=self._episode_weather_seed)
                self._run_warmup(warmup_target=warmup_target)
        else:
            # No pool — full warmup (same as parent)
            self._launch_sitl()
            self._launch_leader(wind_spd, wind_dir, wind_turb,
                                weather_seed=self._episode_weather_seed)
            self._run_warmup(warmup_target=warmup_target)

        # D2: per-episode GPS degradation (v2 OU ladder / v1 legacy write).
        self._apply_gps_degradation(gps_accuracy, gps_dropout_enabled, gps_dropout_max)

        # Create / reset DRL components (pass curriculum stage for correct reward weights)
        self._init_components(stage=self._curriculum_stage)

        # Seed geometry
        self._compute_geometry()

        # Record episode start
        self._episode_start_sim_s = self._time_auth.now_s()
        self._tick_count = 0
        self._continuous_m3b_dwell_s = 0.0
        self._accumulated_m3b_dwell_s = 0.0
        self._milestone_tier = 0
        self._tier_first_achieved_s = {}
        self._tier_dwell_s = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        self._min_range_m = float('inf')
        self._max_behind_overshoot_m = 0.0
        self._max_cross_excursion_m = 0.0
        self._prev_applied_residual = np.zeros(3, dtype=np.float32)
        self._prev_range_m = 0.0

        obs = self._build_observation(dt_s=config.TICK_DT_S)
        info: Dict[str, Any] = {"warmup_complete": True, "from_pool": self._pool is not None}

        # v7: Reset idle-drift counters and pause SITL (mirrors parent reset)
        self._pause_count = 0
        self._unpause_count = 0
        self._max_pause_wall_s = 0.0
        self._pause_wall_start = None
        self._pause_sitl()

        return obs, info

    def _adopt_warmup_result(self, result: WarmupResult) -> None:
        """Take ownership of all handles from a WarmupResult."""
        # MAVLink connection
        self._mav = result.mav

        # Time authority
        self._time_auth = result.time_auth
        if self._time_auth is not None:
            self._time_auth.drl_mode = True

        # Geometry calculator
        self._geom_calc = result.geom_calc

        # Control laws
        self._control_laws = result.control_laws

        # Leader telemetry receiver
        self._leader_rx = result.leader_rx

        # v6 Fix: Rebind sim_now_s_fn to DRL-phase time authority.
        # Without this, control_laws retains the warmup controller's frozen
        # tick_time_s() → dt=0 → slew lock (same bug as baseline path).
        if self._control_laws is not None and self._time_auth is not None:
            self._control_laws.sim_now_s_fn = self._time_auth.now_s

        # Home altitude
        self._home_alt_msl = result.home_alt_msl

        # Subprocess handles (for cleanup)
        self._sitl_proc = result.sitl_proc
        self._leader_proc = result.leader_proc
        self._leader_log_file = result.leader_log_file

        # Port offset and derived ports (must match for orphan cleanup
        # and for the fallback path if pool times out)
        self._port_offset = result.port_offset
        self._follower_port = 14551 + self._port_offset
        self._leader_port = 14550 + self._port_offset
        self._timesync_port = 14553 + self._port_offset

        # Follower telemetry
        self._follower_lat = result.follower_lat
        self._follower_lon = result.follower_lon
        self._follower_vn = result.follower_vn
        self._follower_ve = result.follower_ve
        self._follower_state = FollowerState(
            airspeed_ms=result.follower_airspeed_ms,
            alt_msl_m=result.follower_alt_msl_m,
            heading_deg=result.follower_heading_deg,
            pitch_deg=result.follower_pitch_deg,
            roll_deg=result.follower_roll_deg,
            vd_ms=result.follower_vd_ms,
        )

        # Leader telemetry
        self._leader_lat = result.leader_lat
        self._leader_lon = result.leader_lon
        self._leader_vn = result.leader_vn
        self._leader_ve = result.leader_ve
        self._leader_state = LeaderState(
            airspeed_ms=result.leader_airspeed_ms,
            heading_deg=result.leader_heading_deg,
            alt_msl_m=result.leader_alt_msl_m,
            vd_ms=result.leader_vd_ms,
        )

        self._acquired_result = result
