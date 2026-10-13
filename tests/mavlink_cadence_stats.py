# mavlink_cadence_stats.py
# FlightDock v9.10 — Shared cadence stats helper (ATTITUDE rate / timebase instrumentation)

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple


@dataclass(frozen=True)
class CadencePassCriteria:
    """
    Default thresholds mirror the current v9.10 attitude-rate canary:
      - min_hz=18.0
      - max_dt_p95_s=0.08
      - max_stale_gaps=0
    """

    min_hz: float = 18.0
    max_dt_p95_s: float = 0.08
    max_stale_gaps: int = 0
    min_window_hz: float = 18.0


class MavlinkCadenceStats:
    """
    Measure MAVLink message cadence using MAVLink time_boot_ms (sim-time).

    Intended use:
      * Attitude-rate preflight canary (30-60s window)
      * Harness preflight gate ("do not run validation unless rate is proven")
      * Optional runtime watchdog for mid-run reverts

    Notes:
      * Uses time_boot_ms deltas (ms) -> dt_s (seconds).
      * Tracks dt median/p95, max gap, stale-gap count (4 Hz detector), and windowed min Hz.
      * Percentile convention intentionally matches legacy code:
            idx = min(n-1, int(n*p))
        to avoid metric drift between components.
    """

    def __init__(
        self,
        *,
        stale_gap_thresh_s: float = 0.2,
        window_s: float = 5.0,
        max_dt_samples: Optional[int] = None,
        reset_on_backward: bool = True,
    ) -> None:
        self.stale_gap_thresh_s = float(stale_gap_thresh_s)
        self.window_s = float(window_s)
        self.max_dt_samples = max_dt_samples
        self.reset_on_backward = bool(reset_on_backward)

        # Core counters/state
        self._count: int = 0
        self._first_ms: Optional[int] = None
        self._last_ms: Optional[int] = None

        # dt stats storage (seconds between accepted samples)
        self._dt_samples: Deque[float] = (
            deque(maxlen=max_dt_samples) if max_dt_samples else deque()
        )

        # gap/stale tracking
        self._gap_max_s: float = 0.0
        self._stale_gaps: int = 0

        # anomaly tracking (not cleared on epoch reset unless explicitly requested)
        self._duplicates: int = 0
        self._backward: int = 0
        self._resets: int = 0

        # Sliding-window Hz tracking
        self._window_ms: Deque[int] = deque()
        self._window_hz_last: float = 0.0
        self._min_window_hz: Optional[float] = None

    def reset(self, *, preserve_anomalies: bool = False) -> None:
        """
        Reset accumulated cadence stats.

        Args:
            preserve_anomalies: if True, keep duplicate/backward/reset counters
                               (useful when treating backward time_boot_ms as an epoch reset).
        """
        self._count = 0
        self._first_ms = None
        self._last_ms = None
        self._dt_samples.clear()
        self._gap_max_s = 0.0
        self._stale_gaps = 0
        self._window_ms.clear()
        self._window_hz_last = 0.0
        self._min_window_hz = None

        if not preserve_anomalies:
            self._duplicates = 0
            self._backward = 0
            self._resets = 0

    def reset_startup_counters(self) -> None:
        """Reset counters inflated by startup transients.

        Call after settle gate passes so that gap_max_s and stale_gaps
        reflect only steady-state behavior, not startup artifacts.
        Unlike reset(), this preserves dt_samples, window state, and counts.
        """
        self._gap_max_s = 0.0
        self._stale_gaps = 0

    def observe_time_boot_ms(self, ms: int) -> str:
        """
        Observe a new message timestamp.

        Returns an event tag:
          - "OK" for accepted, monotonic sample
          - "DUPLICATE" for ms == last_ms
          - "BACKWARD" for ms < last_ms (and not reset)
          - "RESET" for backward time that triggers reset_on_backward policy
        """
        ms_i = int(ms)

        if self._first_ms is None:
            # First sample
            self._first_ms = ms_i
            self._last_ms = ms_i
            self._count = 1
            self._window_ms.append(ms_i)
            return "OK"

        assert self._last_ms is not None

        if ms_i == self._last_ms:
            self._duplicates += 1
            return "DUPLICATE"

        if ms_i < self._last_ms:
            self._backward += 1
            if self.reset_on_backward:
                # Treat as epoch reset (reboot/reconnect). Preserve anomaly counters.
                self._resets += 1
                self.reset(preserve_anomalies=True)

                # Re-ingest as first sample of new epoch
                self._first_ms = ms_i
                self._last_ms = ms_i
                self._count = 1
                self._window_ms.append(ms_i)
                return "RESET"
            return "BACKWARD"

        # Monotonic advance
        dt_s = (ms_i - self._last_ms) / 1000.0
        self._last_ms = ms_i
        self._count += 1

        # dt stats
        self._dt_samples.append(dt_s)

        # gap/stale
        if dt_s > self._gap_max_s:
            self._gap_max_s = dt_s
        if dt_s > self.stale_gap_thresh_s:
            self._stale_gaps += 1

        # windowed Hz
        self._window_ms.append(ms_i)
        self._trim_window(ms_i)
        self._window_hz_last = self._compute_window_hz()
        if self._window_hz_last > 0.0:
            if self._min_window_hz is None or self._window_hz_last < self._min_window_hz:
                self._min_window_hz = self._window_hz_last

        return "OK"

    def _trim_window(self, newest_ms: int) -> None:
        """Drop samples older than window_s behind newest_ms."""
        horizon_ms = newest_ms - int(self.window_s * 1000.0)
        while self._window_ms and self._window_ms[0] < horizon_ms:
            self._window_ms.popleft()

    def _compute_window_hz(self) -> float:
        if len(self._window_ms) < 2:
            return 0.0
        dt_s = (self._window_ms[-1] - self._window_ms[0]) / 1000.0
        if dt_s <= 0.0:
            return 0.0
        return (len(self._window_ms) - 1) / dt_s

    @staticmethod
    def _percentile_floor(sorted_vals: list[float], p: float) -> float:
        """Legacy percentile convention: idx = min(n-1, int(n*p))."""
        n = len(sorted_vals)
        if n == 0:
            return 0.0
        idx = int(n * p)
        if idx > n - 1:
            idx = n - 1
        return sorted_vals[idx]

    def dt_stats(self) -> Tuple[float, float, float, float]:
        """Return (min, median, p95, max) of dt samples in seconds."""
        if not self._dt_samples:
            return 0.0, 0.0, 0.0, 0.0
        vals = list(self._dt_samples)
        vals.sort()
        dt_min = vals[0]
        dt_med = statistics.median(vals)
        dt_p95 = self._percentile_floor(vals, 0.95)
        dt_max = vals[-1]
        return dt_min, dt_med, dt_p95, dt_max

    def hz_sim_est(self) -> float:
        """Estimate Hz based on first/last time_boot_ms (sim-time)."""
        if self._first_ms is None or self._last_ms is None or self._count < 2:
            return 0.0
        dt_s = (self._last_ms - self._first_ms) / 1000.0
        if dt_s <= 0.0:
            return 0.0
        return (self._count - 1) / dt_s

    def min_window_hz(self) -> float:
        """Minimum computed sliding-window Hz observed so far (0.0 if not enough data)."""
        return float(self._min_window_hz or 0.0)

    def metrics(self) -> Dict[str, object]:
        """
        Return a stable dict for JSON emission.
        Callers may rename keys (e.g., attitude_rx_*).
        """
        dt_min, dt_med, dt_p95, dt_max = self.dt_stats()
        return {
            "count_total": self._count,
            "time_boot_ms_first": self._first_ms,
            "time_boot_ms_last": self._last_ms,
            "hz_sim_est": self.hz_sim_est(),
            "dt_median_s": dt_med,
            "dt_p95_s": dt_p95,
            "dt_min_s": dt_min,
            "dt_max_s": dt_max,
            "gap_max_s": self._gap_max_s,
            "stale_gaps": self._stale_gaps,
            "stale_gap_thresh_s": self.stale_gap_thresh_s,
            "window_s": self.window_s,
            "window_hz_last": self._window_hz_last,
            "min_window_hz": self.min_window_hz(),
            "duplicates": self._duplicates,
            "backward": self._backward,
            "resets": self._resets,
        }

    def check_pass(self, criteria: CadencePassCriteria) -> Tuple[bool, Dict[str, bool]]:
        """
        Return (overall_pass, per_check_results).

        Intended for canary JSON `pass_results`, matching your current schema:
          hz, dt_p95, stale_gaps, plus window stability via min_window_hz.
        """
        _, _, dt_p95, _ = self.dt_stats()
        hz = self.hz_sim_est()
        min_w_hz = self.min_window_hz()

        results = {
            "hz": hz >= criteria.min_hz,
            "min_window_hz": min_w_hz >= criteria.min_window_hz,
            "dt_p95": dt_p95 <= criteria.max_dt_p95_s,
            "stale_gaps": self._stale_gaps <= criteria.max_stale_gaps,
        }
        overall = all(results.values())
        return overall, results


# --- Optional: mapping helper (example) ---
def map_to_attitude_rx_schema(stats: MavlinkCadenceStats) -> Dict[str, object]:
    """Map generic cadence metrics to the existing attitude_rate_canary_report_* schema."""
    m = stats.metrics()
    return {
        "attitude_rx_count_total": m["count_total"],
        "attitude_rx_time_boot_ms_first": m["time_boot_ms_first"],
        "attitude_rx_time_boot_ms_last": m["time_boot_ms_last"],
        "attitude_rx_hz_sim_est": m["hz_sim_est"],
        "attitude_rx_dt_median_s": m["dt_median_s"],
        "attitude_rx_dt_p95_s": m["dt_p95_s"],
        "attitude_rx_gap_max_s": m["gap_max_s"],
        # keep naming consistent with your current JSON
        "attitude_rx_stale_gaps_over_0p2s_count": m["stale_gaps"],
        "attitude_rx_min_window_hz": m["min_window_hz"],
    }
