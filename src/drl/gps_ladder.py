"""GPS-degradation generators for the FD_CURRICULUM_V2 ladder.

RESPONSIBILITY
    Pure-math, seeded generators that produce the *inputs* for the env-side GPS
    degradation ladder (D2). They own NO MAVLink/SITL handle and perform NO I/O:
    the env applies their outputs via ``_set_sim_param``. EKF3 sits in front of
    the docking fuser, so degraded GPS that reaches the estimate is bias/drift —
    hence a SLOW horizontal GLTCH random-walk (survives EKF3 as drift) plus
    occasional dropout windows. Scatter is intentionally NOT modelled (EKF3
    filters it; see the D2 plan).

PUBLIC API
    GpsOuWalk(cep_m, tau_s, dt_s, rng).step() -> (gltch_x_deg, gltch_y_deg)
    build_dropout_schedule(enabled, max_events, episode_s, rng,
                           min_s, max_dur_s) -> list[(start_s, end_s)]

SIDE EFFECTS
    None. No I/O, no globals, no SITL. The numpy Generator is injected by the
    caller (determinism is the caller's responsibility — see D1 seeding).

CAPS
    file soft 200 / hard 300 LoC; function cap 40 LoC.

COVERAGE
    tests/unit/drl/test_gps_ladder.py (determinism, sigma/median, autocorrelation,
    dropout bounds) — required for D2 GREEN.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from . import config


class GpsOuWalk:
    """Per-axis Ornstein-Uhlenbeck horizontal GPS bias walk.

    x_{k+1} = a*x_k + b*N(0,1), a = exp(-dt/tau), b = sigma*sqrt(1 - a^2).
    Stationary per-axis std = sigma; the 2-D magnitude median ~= sigma*1.1774 =
    CEP. tau (10-30 s) keeps the walk SLOW so it survives EKF3 as drift, not the
    high-frequency jitter EKF3 would filter out. ``step`` returns the lat/lng
    GLTCH offsets in DEGREES (meters * GPS_DEG_PER_M) for SIM_GPS*_GLTCH_X/Y.
    """

    def __init__(self, cep_m: float, tau_s: float, dt_s: float,
                 rng: "np.random.Generator") -> None:
        if rng is None:
            raise ValueError("GpsOuWalk requires a seeded np.random.Generator")
        self._sigma = float(cep_m) / config.GPS_OU_SIGMA_FACTOR  # per-axis meters
        tau_s = max(float(tau_s), 1e-6)
        self._a = math.exp(-float(dt_s) / tau_s)
        self._b = self._sigma * math.sqrt(max(1.0 - self._a * self._a, 0.0))
        self._rng = rng
        # Start at the stationary distribution so episode start isn't pinned to 0.
        self._x = float(self._rng.normal(0.0, self._sigma))
        self._y = float(self._rng.normal(0.0, self._sigma))

    def step(self) -> Tuple[float, float]:
        """Advance one GPS tick; return (gltch_x_deg, gltch_y_deg)."""
        self._x = self._a * self._x + self._b * float(self._rng.normal())
        self._y = self._a * self._y + self._b * float(self._rng.normal())
        return (self._x * config.GPS_DEG_PER_M, self._y * config.GPS_DEG_PER_M)


def build_dropout_schedule(
    enabled: bool,
    max_events: int,
    episode_s: float,
    rng: "np.random.Generator",
    min_s: float = None,
    max_dur_s: float = None,
) -> List[Tuple[float, float]]:
    """Seeded list of (start_s, end_s) GPS-dropout windows, sorted, non-empty
    only when enabled and max_events>0. All times within [0, episode_s]."""
    if not enabled or max_events <= 0 or rng is None:
        return []
    min_s = config.GPS_DROPOUT_MIN_S if min_s is None else min_s
    max_dur_s = config.GPS_DROPOUT_MAX_S if max_dur_s is None else max_dur_s
    n = int(rng.integers(0, int(max_events) + 1))  # 0..max_events
    events: List[Tuple[float, float]] = []
    latest_start = max(float(episode_s) - max_dur_s, 1.0)
    for _ in range(n):
        start = float(rng.uniform(0.0, latest_start))
        dur = float(rng.uniform(min_s, max_dur_s))
        events.append((start, start + dur))
    return sorted(events)
