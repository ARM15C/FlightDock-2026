#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: T-403 UWB gate sidecar. ONE CSV row per UWB update ATTEMPT --
#   ACCEPTED AND REJECTED, every anchor -- carrying the innovation-gate
#   arithmetic with the PREDICTION variance H*P*H' logged SEPARATELY from the
#   assumed measurement noise R, so the counterfactual "which gated rejections
#   would have flipped under R' ?" is computable OFFLINE for ANY R', no re-fly:
#       S' = hpht_m2 + R'
#       sigma' = abs(innov_m) / sqrt(S')
#       accept' = not (sigma' > gate_thresh_sigma)
#
# WHY THIS FILE EXISTS (and is not a column-append to uwb_fusion_diag.csv):
#   that file's header AND every row write live inside
#   tests/baseline_formation_controller.py:7691-7782 == R7.1, NEVER MODIFY. So
#   the export is written fuser-side and JOINED -- the same escape hatch, and
#   the same env-dir resolution, as the uwb_channel_iso.csv precedent
#   (uwb_gps_fusion.py:1086-1118). That precedent is ACCEPT-ONLY, so it cannot
#   be reused: the rejections are the entire question. It also closes T-401 Q5
#   (per-anchor accept/reject counters exist for anchor 0 only).
#
# JOIN CONTRACT -- MEASURED, NOT ASSUMED. This file stamps the MEASUREMENT clock
#   (now_s - HARNESS_UWB_LATENCY_S, default 0.05); uwb_fusion_diag.csv stamps
#   the CONTROLLER TICK clock. On a banked leg only 89.3% of sidecar stamps have
#   an exact partner, and only 22.9% of diag rows do. Join AS-OF on
#   (sim_time_s +/- 1 tick, anchor_id) and REPORT THE UNMATCHED FRACTION as a
#   denominator. An exact-equality join silently drops most of the corpus.
#
# GUARANTEES:
#   * NO NEW ENV FLAG. Gated by the EXISTING FD_UWB_FUSION_DIAG=1; placed by the
#     EXISTING FD_UWB_FUSION_DIAG_DIR. Default OFF -> caller branch not entered.
#   * WRITE-ONLY. Never read back into the filter -> lockstep-deterministic.
#   * NEVER RAISES. A gauge that can take down the estimator is worse than no
#     gauge. Every public entry point swallows Exception.
#   * G4-CLEAN BY CONSTRUCTION. Handed only PERCEIVED quantities (the range the
#     fuser received; its own H / P / S / R). Holds no truth reference and
#     imports NOTHING but stdlib -- no FlightDock module at all.
#   * DOES NOT MUTATE its arguments. H and P are read, never written.
# PUBLIC API: UwbGateDiagWriter (.enabled .rows_written .last .log() .close()),
#   from_env(), FILENAME, HEADER, SCHEMA_VERSION.
# SIDE-EFFECTS: creates out_dir and writes+flushes uwb_gate_diag.csv there, ONLY
#   when enabled AND out_dir is non-empty. from_env() reads two EXISTING env
#   vars. No network, no subprocess, no source-tree writes.
# CAPS: file soft 170 / hard 240 code lines; functions <= 45 lines.
# COVERAGE: tests/unit/test_uwb_gate_diag.py -- seven failure modes, each
#   exercised in BOTH directions (a detector that cannot fire is not a detector).
# =============================================================================
import csv
import logging
import os

logger = logging.getLogger(__name__)

FILENAME = "uwb_gate_diag.csv"
SCHEMA_VERSION = 1

# Mirrors uwb_gps_fusion.IDX_X / IDX_VX (:219, :222). NOT imported: uwb_gps_fusion
# imports THIS module, so importing back would be circular. The unit test asserts
# these equal the fuser's own constants, so drift is CAUGHT, not assumed away.
_IDX_BEHIND_POS, _IDX_BEHIND_VEL = 0, 3
_NSTATE = 6

HEADER = [
    # --- join key. sim_time_s ALONE IS NOT A KEY: every anchor in a batch
    # shares one measurement timestamp, so (sim_time_s, anchor_id) is minimal.
    "sim_time_s", "anchor_id", "seq",
    # --- TIER 1: the R' counterfactual. Nothing here may be dropped, and these
    # are written UNROUNDED so the offline identity is exact ==, not a
    # tolerance (E3c's Q0 was burned by a 1e-6 tolerance failing on 7.5% of rows).
    "innov_m", "hpht_m2", "r_m2",
    # --- TIER 2: reproduce the decision the filter ACTUALLY took.
    "s_m2", "gate_sigma", "gate_thresh_sigma", "accepted",
    "meas_range_m", "predicted_range_m",
    # --- TIER 3: size the recovery, and show WHY. pht_* / S' gives the Kalman
    # gain the admitted range would have carried; h_behind -> 0 near dock IS
    # T-401's sign-blindness, visible per row.
    "pht_behind_pos", "pht_behind_vel",
    "p_behind_m2", "p_cross_m2", "p_below_m2",
    "h_behind", "h_cross", "h_below",
]

# One open handle PER PATH per module identity. Two RelativePoseFusion objects
# in one process (replay benches, analysis sweeps) would otherwise EACH open the
# same path in mode 'w' and truncate the other's rows. NOTE: uwb_gps_fusion is
# importable under three module identities in this repo, so this registry is
# per-identity, not truly per-process. Stated, not over-claimed.
_OPEN = {}


class UwbGateDiagWriter:
    """Per-ATTEMPT UWB gate sidecar. Write-only, lazy-open, never raises."""

    def __init__(self, enabled, out_dir, gate_thresh_sigma):
        self.out_dir = out_dir or ""
        self.enabled = bool(enabled) and bool(self.out_dir)
        self.gate_thresh_sigma = float(gate_thresh_sigma)
        self.rows_written = 0
        self.last = None
        self._file = None
        self._writer = None
        if enabled and not self.out_dir:
            # LOUD, not silent: the frozen controller falls back to its leg dir
            # (baseline_formation_controller.py:7683); the fuser has no such
            # handle, so this configuration writes NOTHING at all.
            logger.warning(
                "T-403 gate sidecar: FD_UWB_FUSION_DIAG=1 but "
                "FD_UWB_FUSION_DIAG_DIR is empty -- %s will NOT be written "
                "(no leg-dir fallback exists fuser-side)", FILENAME)

    def _ensure_open(self):
        """Lazy first-row open, mode 'w', shared per path within this module."""
        path = os.path.join(self.out_dir, FILENAME)
        got = _OPEN.get(path)
        if got is None:
            os.makedirs(self.out_dir, exist_ok=True)
            fh = open(path, "w", newline="")
            wr = csv.writer(fh)
            wr.writerow(HEADER)
            got = _OPEN[path] = (fh, wr)
        self._file, self._writer = got

    def log(self, now, anchor_id, meas_range_m, predicted_range_m, innovation,
            H, P, r_m2, s_m2, gate_sigma, accepted):
        """Record ONE update attempt. Accepted and rejected alike.

        H*P*H' is RECOMPUTED here from the caller's H and P with the byte-same
        expression the filter used (uwb_gps_fusion.py:1230) -- deliberately NOT
        derived as s_m2 - r_m2. A derived hpht makes the offline identity
        hpht + r == s a TAUTOLOGY, and a tautology cannot fail. Recomputed, that
        identity is real evidence the gauge read the same P the gate read."""
        if not self.enabled:
            return
        try:
            rng = range(_NSTATE)
            hpht = sum(H[i] * sum(P[i][j] * H[j] for j in rng) for i in rng)
            pht_p = sum(P[_IDX_BEHIND_POS][j] * H[j] for j in rng)
            pht_v = sum(P[_IDX_BEHIND_VEL][j] * H[j] for j in rng)
            row = (
                round(now, 6), int(anchor_id), self.rows_written,
                innovation, hpht, r_m2,
                s_m2, gate_sigma, self.gate_thresh_sigma, int(bool(accepted)),
                round(meas_range_m, 9), round(predicted_range_m, 9),
                pht_p, pht_v,
                P[0][0], P[1][1], P[2][2],
                H[0], H[1], H[2],
            )
            if self._writer is None:
                self._ensure_open()
            self._writer.writerow(row)
            self._file.flush()      # crash-survivable, as the precedent (:1113)
            self.rows_written += 1
            self.last = row
        except Exception:
            pass                    # a gauge must NEVER raise into the estimator

    def close(self):
        """Idempotent. Not required -- per-row flush + process exit suffice."""
        try:
            if self._file is not None:
                _OPEN.pop(os.path.join(self.out_dir, FILENAME), None)
                self._file.close()
        except Exception:
            pass
        self._file = self._writer = None


def from_env(gate_thresh_sigma):
    """Construct from the EXISTING flags ONLY -- no new env var is introduced.

    FD_UWB_FUSION_DIAG gates it; FD_UWB_FUSION_DIAG_DIR places it (the same rung
    uwb_channel_iso.csv falls back to, uwb_gps_fusion.py:603-606), so the
    sidecar lands beside uwb_fusion_diag.csv and joins against it."""
    return UwbGateDiagWriter(
        enabled=os.environ.get("FD_UWB_FUSION_DIAG", "0").strip() == "1",
        out_dir=os.environ.get("FD_UWB_FUSION_DIAG_DIR", "").strip(),
        gate_thresh_sigma=gate_thresh_sigma,
    )
