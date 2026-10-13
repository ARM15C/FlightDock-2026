#!/usr/bin/env python3
"""F-3 (T-199): generator for the H12 conjunction-halt parameters.

Derives `config.H12_LOSS_THRESHOLD` and `config.H12_MIN_RISE_FRAC` from a
banked HEALTHY corpus of offpolicy_diagnostics.csv files (runs whose windows
passed their gates), replicating H12's own windowing exactly:

  - rise statistic: over every sliding 75-episode window (3 x 25, stride 1)
    whose three 25-ep `critic_grad_norm_mean` means are STRICTLY monotone
    increasing, the fractional rise w3/w1 - 1. Healthy monotone rises come
    from the curriculum wind ladder; a real divergence (L9.2: loss cycling
    45-120) rises by 100%+.
  - loss statistic: every 25-episode rolling mean of `critic_loss` (L25).

MARGIN RULE:
  H12_MIN_RISE_FRAC = 3.0 x max healthy monotone rise, rounded UP to 0.01
  H12_LOSS_THRESHOLD = 1.5 x max healthy L25, rounded UP to 5.0

NaN handling matches the RCA: NaN rows are dropped per-column before
windowing (H12 itself skips windows containing NaN — dropping is the
conservative direction for a healthy-corpus MAX).

Usage:
  python tools/h12_calibration.py <opd.csv> [<opd.csv> ...] [--json OUT]

Exit codes: 0 = OK · 3 = IO/usage or corpus yields no evaluable window.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import pandas as pd

WINDOW = 25
N_WINDOWS = 3
RISE_MARGIN = 3.0
RISE_ROUND = 0.01
LOSS_MARGIN = 1.5
LOSS_ROUND = 5.0


def _round_up(value: float, quantum: float) -> float:
    return round(math.ceil(value / quantum - 1e-9) * quantum, 6)


def corpus_stats(csv_paths):
    """(max_monotone_rise_frac, max_l25, n_monotone_windows, n_l25_windows)
    across the corpus. Raises ValueError if no L25 window is evaluable."""
    rises, l25s = [], []
    span = WINDOW * N_WINDOWS
    for path in csv_paths:
        df = pd.read_csv(path)
        grads = df["critic_grad_norm_mean"].dropna().to_numpy()
        losses = df["critic_loss"].dropna().to_numpy()
        for i in range(len(grads) - span + 1):
            w = grads[i:i + span]
            means = [w[j * WINDOW:(j + 1) * WINDOW].mean()
                     for j in range(N_WINDOWS)]
            if all(means[j] < means[j + 1] for j in range(N_WINDOWS - 1)):
                rises.append(means[-1] / means[0] - 1.0)
        for i in range(len(losses) - WINDOW + 1):
            l25s.append(losses[i:i + WINDOW].mean())
    if not l25s:
        raise ValueError("corpus yields no evaluable L25 window")
    return (max(rises) if rises else 0.0, max(l25s), len(rises), len(l25s))


def recommend(csv_paths):
    """Full calibration record for the given healthy corpus."""
    max_rise, max_l25, n_rise, n_l25 = corpus_stats(csv_paths)
    return {
        "corpus": [str(p) for p in csv_paths],
        "max_healthy_monotone_rise_frac": round(max_rise, 6),
        "max_healthy_l25_loss": round(max_l25, 6),
        "n_monotone_windows": n_rise,
        "n_l25_windows": n_l25,
        "rule": (f"rise_floor = {RISE_MARGIN}x max rise up to {RISE_ROUND}; "
                 f"loss_threshold = {LOSS_MARGIN}x max L25 up to {LOSS_ROUND}"),
        "H12_MIN_RISE_FRAC": _round_up(max_rise * RISE_MARGIN, RISE_ROUND),
        "H12_LOSS_THRESHOLD": _round_up(max_l25 * LOSS_MARGIN, LOSS_ROUND),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csvs", nargs="+", help="healthy offpolicy_diagnostics.csv paths")
    ap.add_argument("--json", default="", help="also write the record to this path")
    args = ap.parse_args(argv)
    try:
        rec = recommend(args.csvs)
    except (OSError, KeyError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(rec, indent=1))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rec, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
