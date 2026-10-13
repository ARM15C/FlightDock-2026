#!/usr/bin/env python3
"""
pvt_disparity_census.py — CANONICAL perception-vs-truth disparity estimator (Estimator B).
===========================================================================================
RESPONSIBILITY Compute matched-timestamp P-T disparities (behind/below/cross) per FSM
  state and time-resolved within DOCKING_CREEP, using the CANONICAL estimator definition
  fixed by (supersedes ad-hoc per-report estimators):
    * sampling frame: TRUTH-LEFT — one sample per DISTINCT truth sim_time_s point
      (the truth observer stamps ~10 Hz wall-clock rows onto a ~0.25 s-quantized sim-time
      grid, ~2.5 rows/point; bursts are collapsed to their per-point MEDIAN to reduce
      stamping smear). n reported = distinct truth time points, NOT raw rows.
    * match: pandas merge_asof nearest, tolerance 0.06 s, against the ~20 Hz perceived logs.
    * perceived consumed = speed_arbitration.csv behind_m/below_m/cross_m (verified
      identical to diag behind_gps + d_corr delta within 0.0002 m on sigclean leg);
      perceived GPS-only = uwb_fusion_diag.csv behind_gps/below_gps.
    * quantiles: SIGNED med/p5/p95/p99; |.| p95 and |.| p99 additionally; plus RMS
      and SD (grade_retune f8 is |.|). p99+RMS added 2026-06-27 for M3b analyses.
  MEASUREMENT FLOOR (carry with every quote): within-burst truth spread med ~0.12 m /
  p95 ~0.28 m (behind, sigclean DC) from the 0.25 s sim-time quantization — inflates
  spread metrics (sd/p95), leaves medians ~unbiased. FOLLOWER_TRUTH_FEED removes this.
PUBLIC API CLI: --leg-dir [--dc-bins N] [--out-json PATH]; __main__-guarded.
SIDE EFFECTS reads leg artifacts (read-only); writes optional JSON + stdout.
CAPS soft 220 / hard 260 lines; functions <= 35 LoC.
COVERAGE py_compile; numbers cross-checked against the sigclean reconciliation
  (analysis/sigclean_20260612T193630Z/analysis/pvt_estimator_reconciliation.py).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

TOL_S = 0.06
DC = "DOCKING_CREEP"
MIN_STATE_N = 30


def load_truth_dedup(leg):
    """Truth log collapsed to one median row per distinct sim_time_s point."""
    rows = [json.loads(l) for l in open(os.path.join(leg, "paired_tick_log_env0.jsonl"))]
    tr = pd.DataFrame(rows)
    cols = ["raw_truth_behind_m", "raw_truth_below_m", "raw_truth_cross_m"]
    g = tr.groupby("sim_time_s", as_index=False)[cols].median()
    burst = tr.groupby("sim_time_s")["raw_truth_behind_m"].agg(lambda s: s.max() - s.min())
    meta = {"truth_rows_raw": int(len(tr)), "truth_points_distinct": int(len(g)),
            "burst_behind_spread_med_m": round(float(burst.median()), 4),
            "burst_behind_spread_p95_m": round(float(burst.quantile(.95)), 4)}
    return g.sort_values("sim_time_s").reset_index(drop=True), meta


def qrow(d):
    d = d.dropna()
    if len(d) < MIN_STATE_N:
        return None
    return {"n": int(len(d)), "med": round(float(d.median()), 3),
            "p5": round(float(d.quantile(.05)), 3), "p95": round(float(d.quantile(.95)), 3),
            "p99": round(float(d.quantile(.99)), 3),
            "abs_p95": round(float(d.abs().quantile(.95)), 3),
            "abs_p99": round(float(d.abs().quantile(.99)), 3),
            "rms": round(float(np.sqrt((d ** 2).mean())), 3),
            "sd": round(float(d.std()), 3)}


def match(tr, right, rcols, state_col):
    m = pd.merge_asof(tr, right[["sim_time_s", state_col] + rcols].sort_values("sim_time_s"),
                      on="sim_time_s", tolerance=TOL_S, direction="nearest")
    return m.dropna(subset=[state_col])


def per_state(m, pairs, state_col):
    out = {}
    for st, g in m.groupby(state_col):
        row = {}
        for label, (pc, tc) in pairs.items():
            r = qrow(g[pc] - g[tc])
            if r:
                row[label] = r
        if row:
            row["t0_s"] = round(float(g.sim_time_s.iloc[0]), 1)
            row["t1_s"] = round(float(g.sim_time_s.iloc[-1]), 1)
            out[st] = row
    return out


def dc_bins(m, pairs, state_col, nbins):
    dc = m[m[state_col] == DC].copy()
    if len(dc) < MIN_STATE_N:
        return {}
    dc["bin"] = pd.cut(dc.sim_time_s, nbins)
    out = {}
    for b, g in dc.groupby("bin", observed=True):
        row = {}
        for label, (pc, tc) in pairs.items():
            r = qrow(g[pc] - g[tc])
            if r:
                row[label] = {"med": r["med"], "p95": r["p95"]}
        out[f"{b.left:.0f}-{b.right:.0f}s"] = row
    drift = {}
    t_min = (dc.sim_time_s - dc.sim_time_s.min()) / 60.0
    for label, (pc, tc) in pairs.items():
        d = dc[pc] - dc[tc]
        ok = d.notna()
        if ok.sum() > MIN_STATE_N:
            drift[label] = round(float(np.polyfit(t_min[ok], d[ok], 1)[0] * 1000), 1)
    return {"bins": out, "drift_mm_per_min": drift}


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--leg-dir", required=True)
    pa.add_argument("--dc-bins", type=int, default=8)
    pa.add_argument("--out-json")
    a = pa.parse_args()

    tr, meta = load_truth_dedup(a.leg_dir)
    sa = pd.read_csv(os.path.join(a.leg_dir, "speed_arbitration.csv"), low_memory=False)
    dg = pd.read_csv(os.path.join(a.leg_dir, "uwb_fusion_diag.csv"), low_memory=False)

    cons_pairs = {"behind_consumed": ("behind_m", "raw_truth_behind_m"),
                  "below_consumed": ("below_m", "raw_truth_below_m"),
                  "cross_consumed": ("cross_m", "raw_truth_cross_m")}
    gps_pairs = {"behind_gps": ("behind_gps", "raw_truth_behind_m"),
                 "below_gps": ("below_gps", "raw_truth_below_m")}

    mc = match(tr, sa, ["behind_m", "below_m", "cross_m"], "state")
    mg = match(tr, dg, ["behind_gps", "below_gps"], "controller_state")

    R = {"estimator": "B-canonical (truth-left, distinct-sim_time median-collapsed, tol 0.06 s)",
         "leg_dir": a.leg_dir, "meta": meta,
         "consumed_by_state": per_state(mc, cons_pairs, "state"),
         "gps_by_state": per_state(mg, gps_pairs, "controller_state"),
         "dc_time_resolved": dc_bins(mc, cons_pairs, "state", a.dc_bins)}

    print(f"== P-T DISPARITY CENSUS (Estimator B canonical) — {a.leg_dir}")
    print(f"   truth rows {meta['truth_rows_raw']} -> distinct points {meta['truth_points_distinct']}; "
          f"burst spread med/p95 {meta['burst_behind_spread_med_m']}/{meta['burst_behind_spread_p95_m']} m")
    for section in ("consumed_by_state", "gps_by_state"):
        print(f"-- {section}")
        for st, row in R[section].items():
            print(f"  [{st}] t={row['t0_s']}-{row['t1_s']}s")
            for k, v in row.items():
                if isinstance(v, dict):
                    print(f"     {k:16s} n={v['n']:5d} med={v['med']:+.3f} "
                          f"p95={v['p95']:+.3f} |.|p95={v['abs_p95']:.3f} "
                          f"|.|p99={v['abs_p99']:.3f} rms={v['rms']:.3f}")
    if R["dc_time_resolved"]:
        print("-- DC time-resolved (med/p95 per bin)")
        for b, row in R["dc_time_resolved"]["bins"].items():
            cells = "  ".join(f"{k.split('_')[0]} {v['med']:+.3f}/{v['p95']:+.3f}"
                              for k, v in row.items())
            print(f"  {b:12s} {cells}")
        print(f"  drift mm/min: {R['dc_time_resolved']['drift_mm_per_min']}")
    if a.out_json:
        json.dump(R, open(a.out_json, "w"), indent=2, default=str)
        print(f"json -> {a.out_json}")


if __name__ == "__main__":
    main()
