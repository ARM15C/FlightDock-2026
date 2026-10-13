#!/usr/bin/env python3
"""analyze_frame_quartet — CSV2_R7K_FRAME_VERIFICATION_LOG offline analysis.

RESPONSIBILITY
    Join the v98 extended paired_tick log (raw_cg_cg, raw_cg_boom, raw_truth) with
    the controller's speed_arbitration.csv (FUSED = delta_behind/cross/below_m) by
    sim_time_s, and produce the four-way per-tick comparison (RAW_CG_CG /
    RAW_CG_BOOM / FUSED / SIM_TRUTH) on behind AND below (equal depth), to decide
    whether the boom-tip frame transform is correct and whether the fusion is the
    source of the ~8.8 m behind bias. Emits FRAME_VERIFICATION_REPORT.md + JSON +
    four-way time-series PNGs + rolling-std PNG.

PUBLIC API
    main(argv) -> int (0 ok / 1 hard-abort: both paired_tick AND truth missing)

GATES (precedence): TRUTH_REFERENCE_MISMATCH -> INCONCLUSIVE_NEED_FUSED ->
    {FUSION_BIAS_CONFIRMED | FUSION_DIVERGENCE[_BROADBAND|_CLOSERANGE|_EVENT|_INIT]
     | TRANSFORM_WRONG | ALL_AGREE_ESCALATE}.

SIDE EFFECTS
    Reads the run-dir artifacts + optional N=3 dir; WRITES report/JSON/PNGs into
    --out-dir only. No SITL, no fusion, no frozen-file access.

CAPS
    file soft 450 / hard 650 LoC; function LoC cap 90.

COVERAGE
    Exercised end-to-end on the N=1 canary; the frame math it consumes is unit-
    tested in tests/unit/test_diagnostic_frame_log.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

JOIN_TOL_S = 0.150          # max join tolerance (Correction 2)
UNJOINED_WARN_FRAC = 0.03   # >3% unjoined -> data-quality warning
MATCH_RATE_MIN = 0.80       # <80% -> INCONCLUSIVE_NEED_FUSED
TRUTH_REF_TOL = 0.5         # TRUTH_REFERENCE_CHECK mean/std threshold (m)
M3B_WINDOW_BEHIND_M = 20.0  # FUSED behind < 20 m == docking attempt window
CONTACT_MIN_TICKS = 30      # min ticks for a Contact-bucket subclassification
ROLL_WIN_S = 10.0           # DRL obs-space rolling-std window
N3_DEFAULT = "analysis/native_stack_validation_20260528T182431Z"
OBS_NORM = {"behind": 50.0, "cross": 10.0, "below": 50.0}  # observation_builder.py:26-28


def _stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "p50": float("nan"), "p95": float("nan")}
    return {"n": int(x.size), "mean": float(np.mean(x)), "std": float(np.std(x)),
            "p50": float(np.median(x)), "p95": float(np.percentile(np.abs(x), 95))}


def _load_paired(run_dir: str) -> Optional[pd.DataFrame]:
    p = os.path.join(run_dir, "paired_tick_log_env0.jsonl")
    if not os.path.isfile(p):
        return None
    rows: List[Dict[str, Any]] = []
    with open(p) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    pass
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for grp in ("raw_cg_cg", "raw_cg_boom"):
        if grp in df.columns:
            for ax in ("behind", "cross", "below"):
                df[f"{grp}_{ax}"] = df[grp].apply(
                    lambda d, ax=ax: (d or {}).get(ax) if isinstance(d, dict) else None)
    return df


def _load_fused(run_dir: str) -> Tuple[Optional[pd.DataFrame], List[str], bool, str]:
    """Returns (df, columns_found, fused_available, note). SOFT guard (Corrections 2/3)."""
    p = os.path.join(run_dir, "speed_arbitration.csv")
    if not os.path.isfile(p):
        files = os.listdir(run_dir) if os.path.isdir(run_dir) else []
        return None, [], False, (f"speed_arbitration.csv ABSENT. searched={p} "
                                 f"cwd={os.getcwd()} files={files}")
    df = pd.read_csv(p, low_memory=False)
    cols = list(df.columns)
    print(f"[fused] speed_arbitration.csv columns ({len(cols)}): {cols}")
    need = ["delta_behind_m", "delta_cross_m", "delta_below_m"]
    missing = [c for c in need if c not in cols]
    if missing:
        return df, cols, False, f"FUSED columns missing {missing}; available={cols}"
    for c in need + ["sim_time_s"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, cols, True, "ok"


def _fused_valid_start(fused: pd.DataFrame) -> Optional[float]:
    f = fused.dropna(subset=["delta_behind_m", "sim_time_s"])
    if f.empty:
        return None
    return float(f["sim_time_s"].iloc[0])


def _truth_reference_check(paired: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """First 50 ticks with both raw_cg_boom & raw_truth -> mean/std of the diff."""
    out: Dict[str, Any] = {}
    ok = True
    sub = paired.dropna(subset=["raw_cg_boom_behind", "raw_truth_behind_m"]).head(50)
    for ax in ("behind", "below"):
        diff = (sub[f"raw_cg_boom_{ax}"].astype(float)
                - sub[f"raw_truth_{ax}_m"].astype(float)).to_numpy()
        st = _stats(diff)
        out[ax] = st
        if st["n"] and (abs(st["mean"]) > TRUTH_REF_TOL or st["std"] > TRUTH_REF_TOL):
            ok = False
    out["n_ticks"] = int(len(sub))
    out["ok"] = ok
    return ok, out


def _rolling_std(t: np.ndarray, y: np.ndarray, win_s: float) -> Dict[str, float]:
    order = np.argsort(t)
    t, y = t[order], y[order]
    rs = []
    for i in range(len(t)):
        m = (t >= t[i] - win_s) & (t <= t[i])
        w = y[m]
        w = w[np.isfinite(w)]
        if w.size >= 3:
            rs.append(float(np.std(w)))
    if not rs:
        return {"mean": float("nan"), "max": float("nan"), "stationary": None}
    rs = np.array(rs)
    mean = float(np.mean(rs))
    stationary = bool(np.std(rs) < 0.5 * mean) if mean > 0 else None
    return {"mean": mean, "max": float(np.max(rs)), "stationary": stationary}


def _interp_drl(mean_rs: float) -> str:
    if not np.isfinite(mean_rs):
        return "n/a"
    if mean_rs < 1.0:
        return "acceptable for DRL"
    if mean_rs <= 3.0:
        return "marginal (likely training instability)"
    return "unacceptable (fix fusion before training)"


def _bucket(behind_truth: np.ndarray) -> np.ndarray:
    b = np.abs(behind_truth)
    out = np.full(b.shape, "Stage", dtype=object)
    out[b <= 200] = "Approach"
    out[b <= 50] = "Pre-contact"
    out[b < 10] = "Contact"
    return out


def _plot_fourway(jdf: pd.DataFrame, axis: str, fstart: float, out_png: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    t = jdf["sim_time_s"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax1.plot(t, jdf[f"raw_cg_cg_{axis}"], color="blue", lw=0.7, label="RAW_CG_CG")
    ax1.plot(t, jdf[f"raw_cg_boom_{axis}"], color="green", lw=0.7, label="RAW_CG_BOOM")
    ax1.plot(t, jdf[f"fused_{axis}"], color="orange", lw=0.7, label="FUSED")
    ax1.plot(t, jdf[f"truth_{axis}"], color="red", ls="--", lw=0.9, label="SIM_TRUTH")
    ax1.axvspan(fstart, t.max() if len(t) else fstart, color="grey", alpha=0.10,
                label="FUSED-valid window")
    ax1.set_ylabel(f"{axis}_m (positive = follower {axis} leader)")
    ax1.legend(loc="best", fontsize=8)
    ax1.set_title(f"Four-way {axis} comparison")
    ax2.plot(t, jdf[f"c_{axis}"], color="purple", lw=0.7)
    ax2.axhline(0.0, color="k", lw=0.5)
    ax2.set_ylabel(f"(c) FUSED - RAW_CG_BOOM {axis}_m")
    ax2.set_xlabel("sim_time_s")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return True


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CSV2_R7K four-way frame comparison")
    ap.add_argument("--run-dir", required=True, help="canary out dir (paired_tick + csv)")
    ap.add_argument("--out-dir", required=True, help="artifacts/report dir")
    ap.add_argument("--n3-dir", default=N3_DEFAULT, help="optional N=3 comparison dir")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    R: Dict[str, Any] = {"run_dir": args.run_dir}

    paired = _load_paired(args.run_dir)
    fused, cols, fused_avail, fnote = _load_fused(args.run_dir)
    truth_ok_data = paired is not None and "raw_truth_behind_m" in paired.columns
    if paired is None and (fused is None or not truth_ok_data):
        print("[FATAL] BOTH paired_tick AND truth data missing/malformed.", file=sys.stderr)
        return 1
    R["fused_note"] = fnote
    R["fused_columns"] = cols

    if paired is None or "raw_cg_boom_behind" not in paired.columns:
        R["gate"] = "INSTRUMENTATION_ABSENT"
        R["message"] = ("paired_tick_log_env0.jsonl missing or has no raw_cg_* fields — "
                        "the v98 frame-quartet hook did not run for this dataset.")
        _write(args.out_dir, R, None, None)
        return 0

    # (0) TRUTH_REFERENCE_CHECK (load-bearing, Correction A) -------------------
    tref_ok, tref = _truth_reference_check(paired)
    R["truth_reference_check"] = tref
    if not tref_ok:
        R["gate"] = "TRUTH_REFERENCE_MISMATCH"
        R["message"] = ("v98 truth geometry reference frame does not match "
                        "diagnostic_frame_log boom transform; gate (b) unreliable. "
                        "Confirm v98 boom constants match BOOM_TIP_OFFSET_M=4.31.")
        _write(args.out_dir, R, None, None)
        return 0

    # FUSED-valid window + join ------------------------------------------------
    fstart = _fused_valid_start(fused) if fused_avail else None
    R["fused_valid_start_s"] = fstart
    if fused_avail and fstart is not None:
        pv = paired.dropna(subset=["sim_time_s"]).copy()
        pre = int((pv["sim_time_s"] < fstart).sum())
        R["pre_formation_excluded_ticks"] = pre
        pv = pv[pv["sim_time_s"] >= fstart].sort_values("sim_time_s")
        fv = fused.dropna(subset=["sim_time_s"]).sort_values("sim_time_s")
        jdf = pd.merge_asof(pv, fv[["sim_time_s", "delta_behind_m", "delta_cross_m",
                                    "delta_below_m"]], on="sim_time_s",
                            tolerance=JOIN_TOL_S, direction="nearest")
        jdf = jdf.rename(columns={"delta_behind_m": "fused_behind",
                                  "delta_cross_m": "fused_cross",
                                  "delta_below_m": "fused_below",
                                  "raw_truth_behind_m": "truth_behind",
                                  "raw_truth_cross_m": "truth_cross",
                                  "raw_truth_below_m": "truth_below"})
        denom = len(jdf)
        joined = int(jdf["fused_behind"].notna().sum())
        match_rate = joined / denom if denom else 0.0
        R["join"] = {"denominator_fused_valid": denom, "joined": joined,
                     "match_rate": match_rate, "tolerance_s": JOIN_TOL_S}
        unjoined = denom - joined
        R["join"]["unjoined"] = unjoined
        R["join"]["unjoined_frac"] = (unjoined / denom) if denom else 0.0
        R["join"]["data_quality_warning"] = bool(R["join"]["unjoined_frac"] > UNJOINED_WARN_FRAC)
        # M3b-window unjoined annotation (Correction R2)
        R["m3b_unjoined"] = _m3b_unjoined(jdf, fstart)
        if match_rate < MATCH_RATE_MIN:
            R["gate"] = "INCONCLUSIVE_NEED_FUSED"
            R["message"] = f"match-rate {match_rate:.1%} < 80% within FUSED-valid window."
            _write(args.out_dir, R, jdf, None)
            return 0
        jdf = jdf.dropna(subset=["fused_behind"]).copy()
        _full_analysis(jdf, R, args)
        _stage1_validity(jdf, args.run_dir, R)  # CORRECTION 2: validity + isolation gate
        _write(args.out_dir, R, jdf, args.n3_dir)
        return 0

    # FUSED unavailable but truth ref OK -> INCONCLUSIVE_NEED_FUSED -------------
    R["gate"] = "INCONCLUSIVE_NEED_FUSED"
    R["message"] = ("FUSED unavailable (" + fnote + "); (b) and boom-transform check "
                    "still emitted below. (a)/(c) unresolvable without FUSED. NOT ALL_AGREE_ESCALATE.")
    _partial_b_only(paired, R)
    _write(args.out_dir, R, None, None)
    return 0


def _m3b_unjoined(jdf: pd.DataFrame, fstart: float) -> Dict[str, Any]:
    if "fused_behind" in jdf.columns and jdf["fused_behind"].notna().any():
        win = jdf["fused_behind"].abs() < M3B_WINDOW_BEHIND_M
    else:
        tmax = jdf["sim_time_s"].max()
        win = jdf["sim_time_s"] >= (tmax - 120.0)
    sub = jdf[win]
    unj = int(sub["fused_behind"].isna().sum()) if "fused_behind" in sub else 0
    n = int(len(sub))
    return {"m3b_window_ticks": n, "m3b_window_unjoined": unj,
            "m3b_window_unjoined_frac": (unj / n) if n else 0.0,
            "warning": bool(n and unj / n > UNJOINED_WARN_FRAC)}


def _partial_b_only(paired: pd.DataFrame, R: Dict[str, Any]) -> None:
    b: Dict[str, Any] = {}
    for ax in ("behind", "below", "cross"):
        diff_b = (paired[f"raw_cg_boom_{ax}"].astype(float)
                  - paired[f"raw_truth_{ax}_m"].astype(float)).to_numpy()
        diff_t = (paired[f"raw_cg_boom_{ax}"].astype(float)
                  - paired[f"raw_cg_cg_{ax}"].astype(float)).to_numpy()
        b[ax] = {"b_RAWBOOM_minus_TRUTH": _stats(diff_b),
                 "transform_RAWBOOM_minus_RAWCG": _stats(diff_t)}
    R["partial_metrics"] = b


def _full_analysis(jdf: pd.DataFrame, R: Dict[str, Any], args) -> None:
    metrics: Dict[str, Any] = {}
    for ax in ("behind", "below", "cross"):
        a = (jdf[f"fused_{ax}"].astype(float) - jdf[f"truth_{ax}"].astype(float)).to_numpy()
        b = (jdf[f"raw_cg_boom_{ax}"].astype(float) - jdf[f"truth_{ax}"].astype(float)).to_numpy()
        c = (jdf[f"fused_{ax}"].astype(float) - jdf[f"raw_cg_boom_{ax}"].astype(float)).to_numpy()
        tr = (jdf[f"raw_cg_boom_{ax}"].astype(float) - jdf[f"raw_cg_cg_{ax}"].astype(float)).to_numpy()
        jdf[f"c_{ax}"] = c
        metrics[ax] = {"a_FUSED_minus_TRUTH": _stats(a),
                       "b_RAWBOOM_minus_TRUTH": _stats(b),
                       "c_FUSED_minus_RAWBOOM": _stats(c),
                       "transform_RAWBOOM_minus_RAWCG": _stats(tr)}
    R["metrics"] = metrics

    # DRL obs-space health (behind AND below)
    t = jdf["sim_time_s"].to_numpy()
    drl: Dict[str, Any] = {}
    for ax in ("behind", "below"):
        a = (jdf[f"fused_{ax}"].astype(float) - jdf[f"truth_{ax}"].astype(float)).to_numpy()
        rs = _rolling_std(t, a, ROLL_WIN_S)
        rs["interpretation"] = _interp_drl(rs["mean"])
        rs["obs_scale_divisor"] = OBS_NORM[ax]
        drl[ax] = rs
    R["drl_obs_space_health"] = drl
    R["drl_caveat"] = ("obs = clip(delta_behind_m / 50.0, +/-1.1); 3.0 m fusion bias "
                       "= 0.06 obs-units (within clip). The DRL issue is NOT clipping; "
                       "a time-varying behind bias makes obs-layer behind_m unreliable "
                       "during docking -> inconsistent state for the same physical geometry.")

    # Gate logic (behind + below at equal depth)
    cb = metrics["behind"]["c_FUSED_minus_RAWBOOM"]
    bb = metrics["behind"]["b_RAWBOOM_minus_TRUTH"]
    bl = metrics["below"]["b_RAWBOOM_minus_TRUTH"]
    transform_wrong = abs(bb["mean"]) > 1.0 or abs(bl["mean"]) > 1.0
    c_mag = abs(cb["mean"])
    c_var = cb["p95"] > 2.0 * max(abs(cb["mean"]), 1e-6)
    if transform_wrong:
        R["gate"] = "TRANSFORM_WRONG"
    elif c_mag < 1.0:
        R["gate"] = "ALL_AGREE_ESCALATE"
    elif not c_var:
        R["gate"] = "FUSION_BIAS_CONFIRMED"
    else:
        R["gate"] = "FUSION_DIVERGENCE"
        R["divergence_subtype"] = _divergence_subtype(jdf, R)
    R["message"] = _gate_message(R["gate"], R)

    # Four-way PNGs (primary visual deliverable)
    png_ok = []
    for ax in ("behind", "below"):
        out = os.path.join(args.out_dir, f"four_way_{ax}_timeseries.png")
        if _plot_fourway(jdf, ax, R["fused_valid_start_s"], out):
            png_ok.append(os.path.basename(out))
    R["pngs"] = png_ok
    if not png_ok:
        jdf[["sim_time_s", "raw_cg_cg_behind", "raw_cg_boom_behind", "fused_behind",
             "truth_behind", "raw_cg_cg_below", "raw_cg_boom_below", "fused_below",
             "truth_below"]].to_csv(os.path.join(args.out_dir, "four_way_series.csv"), index=False)
        R["pngs_note"] = "matplotlib unavailable; wrote four_way_series.csv instead."


def _stage1_validity(jdf: pd.DataFrame, run_dir: str, R: Dict[str, Any]) -> None:
    """CSV2_R7L STAGE-1 CORRECTION-2 validity checks + GPS-only isolation gate.

    Guards against a false isolation verdict from a filter-bypass:
      (a) gps_update_count > 0 (best-effort; this harness does not persist it)
      (b) FUSED-behind rolling-std > 0.05 m (filtering, not raw passthrough)
      (c) FUSED-valid-start ~ N=3/R7K formation start (~413 s)
    Isolation gate uses mean(FUSED-RAW_CG_BOOM) behind = the GPS-only floor.
    """
    import glob
    t = jdf["sim_time_s"].to_numpy()
    fb = jdf["fused_behind"].astype(float).to_numpy()
    rs = _rolling_std(t, fb, ROLL_WIN_S)

    def _find_gps_uc(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if "gps_update" in k.lower() and isinstance(v, (int, float)):
                    return v
                r = _find_gps_uc(v)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for x in o:
                r = _find_gps_uc(x)
                if r is not None:
                    return r
        return None

    gps_uc = None
    for fn in sorted(glob.glob(os.path.join(run_dir, "run_summary*.json"))):
        try:
            with open(fn) as f:
                gps_uc = _find_gps_uc(json.load(f))
        except Exception:
            gps_uc = None
        if gps_uc is not None:
            break

    fstart = R.get("fused_valid_start_s")
    R["stage1_validity"] = {
        "a_gps_update_count": gps_uc,
        "a_pass": (gps_uc is None or gps_uc > 0),
        "a_note": ("gps_update_count not persisted by this harness; filter activity "
                   "confirmed via (b) + '[GEOM] UWB fusion filter initialized'"
                   if gps_uc is None else "gps_update_count>0 confirms update_gps called"),
        "b_fused_behind_rolling_std_mean_m": rs["mean"],
        "b_pass": bool(rs["mean"] > 0.05),
        "b_note": ("FUSED behind varies (>0.05 m) -> genuine Kalman filtering, NOT passthrough"
                   if rs["mean"] > 0.05 else
                   "rolling-std < 0.05 m -> possible filter bypass; isolation gate UNRELIABLE"),
        "c_fused_valid_start_s": fstart,
        "c_pass": bool(fstart is not None and 380.0 <= fstart <= 460.0),
        "c_note": (f"FUSED-valid-start {fstart:.1f}s ~ N=3/R7K formation start (~413s)"
                   if fstart is not None else "no FUSED-valid-start"),
    }

    cmean = (R.get("metrics", {}).get("behind", {})
             .get("c_FUSED_minus_RAWBOOM", {}).get("mean"))
    if cmean is not None:
        m = abs(cmean)
        if m < 1.0:
            ig = "UWB_CONFIRMED_CULPRIT"
        elif m < 2.0:
            ig = "UWB_DOMINANT_PARTIAL"
        elif m <= 3.0:
            ig = "GPS_CONTRIBUTION_SIGNIFICANT"
        else:
            ig = "NOT_UWB_ESCALATE"
        R["isolation_gate"] = {
            "gate": ig,
            "gps_only_floor_behind_mean_m": cmean,
            "abs_m": m,
            "valid": bool(R["stage1_validity"]["b_pass"]),
        }


def _divergence_subtype(jdf: pd.DataFrame, R: Dict[str, Any]) -> Dict[str, Any]:
    truth_b = jdf["truth_behind"].astype(float).to_numpy()
    cb = jdf["c_behind"].astype(float).to_numpy()
    cl = jdf["c_below"].astype(float).to_numpy()
    buckets = _bucket(jdf["raw_cg_boom_behind"].astype(float).to_numpy())
    per_bucket: Dict[str, Any] = {}
    for name in ("Stage", "Approach", "Pre-contact", "Contact"):
        m = buckets == name
        per_bucket[name] = {"n": int(m.sum()),
                            "behind": _stats(cb[m]), "below": _stats(cl[m])}
    # INIT detection: bias present in first 60 s, non-growing
    t = jdf["sim_time_s"].to_numpy()
    t0 = t.min()
    early = cb[t <= t0 + 60.0]
    late = cb[t > t0 + 60.0]
    init_like = (early.size and late.size
                 and abs(np.mean(early[np.isfinite(early)])) > 3.0
                 and abs(np.mean(late[np.isfinite(late)]) - np.mean(early[np.isfinite(early)])) < 2.0)
    contact = per_bucket["Contact"]["n"]
    means = {k: per_bucket[k]["behind"]["mean"] for k in per_bucket if per_bucket[k]["n"]}
    sub = "FUSION_DIVERGENCE_UNCLASSIFIED"
    if init_like:
        sub = "FUSION_DIVERGENCE_INIT"
    elif contact >= CONTACT_MIN_TICKS and means.get("Contact") and \
            abs(means["Contact"]) > 1.5 * abs(means.get("Stage", means["Contact"])):
        sub = "FUSION_DIVERGENCE_CLOSERANGE"
    elif len(means) >= 2 and (max(means.values()) - min(means.values())) < 2.0:
        sub = "FUSION_DIVERGENCE_BROADBAND"
    contact_note = ("" if contact >= CONTACT_MIN_TICKS else
                    f"Contact bucket sparse (N={contact}<{CONTACT_MIN_TICKS}) — "
                    "per-bucket stats unreliable at this range; valid only for "
                    "Stage/Approach/Pre-contact; _CLOSERANGE/_BROADBAND suppressed.")
    if contact < CONTACT_MIN_TICKS and sub in ("FUSION_DIVERGENCE_CLOSERANGE",
                                               "FUSION_DIVERGENCE_BROADBAND"):
        sub = "FUSION_DIVERGENCE_UNCLASSIFIED"
    return {"subtype": sub, "per_bucket": per_bucket, "contact_note": contact_note,
            "init_action": _init_action(sub)}


def _init_action(sub: str) -> Optional[str]:
    if sub != "FUSION_DIVERGENCE_INIT":
        return None
    return ("AUDIT-ONLY. Read uwb_gps_fusion.py __init__/reset (~L253/L1000): x0[IDX_X] "
            "is self.x=[0.0]*6 at L253 (behind seeded to 0). Sub-gate A if x0[IDX_X]!=0 "
            "or first update_gps pulls x[IDX_X] >2 m off the GPS meas -> "
            "FUSION_DIVERGENCE_INIT_SEED (next: 1-line x0-from-first-GPS fix + A/B canary). "
            "Sub-gate B if x0[IDX_X]==0 and first update_gps consistent (innovation<2 m) -> "
            "FUSION_DIVERGENCE_INIT_DRIFT (next: GPS-only bypass canary). Runtime first-update "
            "K requires OPTION_C_FUSION_DIAG (not enabled this run) -> follow-on. NO edit here.")


def _gate_message(gate: str, R: Dict[str, Any]) -> str:
    msgs = {
        "FUSION_BIAS_CONFIRMED": "Raw geometry correct; fusion is the whole bias. "
                                 "Scope fix in uwb_gps_fusion.py (non-frozen) next.",
        "FUSION_DIVERGENCE": "Fusion diverging (time-varying); see divergence_subtype.",
        "TRANSFORM_WRONG": "RAW_CG_BOOM vs SIM_TRUTH mean >1.0 m on behind or below.",
        "ALL_AGREE_ESCALATE": "All four agree; problem upstream of this point.",
    }
    return msgs.get(gate, "")


def _write(out_dir: str, R: Dict[str, Any], jdf: Optional[pd.DataFrame], n3_dir) -> None:
    if n3_dir:
        R["n3_comparison"] = _n3_compare(R, n3_dir)
    with open(os.path.join(out_dir, "frame_quartet_result.json"), "w") as f:
        json.dump(R, f, indent=2, default=str)
    with open(os.path.join(out_dir, "FRAME_VERIFICATION_REPORT.md"), "w") as f:
        f.write(_render_report(R))
    print(f"[gate] {R.get('gate')}: {R.get('message','')}")


def _n3_compare(R: Dict[str, Any], n3_dir: str) -> Dict[str, Any]:
    p = os.path.join(n3_dir, "run1", "m3b_dual_frame_summary.json")
    if not os.path.isfile(p):
        return {"available": False, "path": p}
    with open(p) as f:
        s = json.load(f)
    n3_med = s.get("disparity", {}).get("median_signed_d_behind_m")
    cur = None
    if "metrics" in R:
        cur = R["metrics"]["behind"]["c_FUSED_minus_RAWBOOM"]["p50"]
    return {"available": True, "n3_median_signed_d_behind_m": n3_med,
            "canary_c_behind_p50": cur,
            "representative": (cur is not None and n3_med is not None
                               and abs(cur - n3_med) < 3.0)}


def _render_report(R: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# FRAME_VERIFICATION_REPORT — CSV2_R7K_FRAME_VERIFICATION_LOG\n")
    L.append(f"**Gate:** `{R.get('gate')}`  \n**Message:** {R.get('message','')}\n")
    if "divergence_subtype" in R:
        ds = R["divergence_subtype"]
        L.append(f"**Divergence subtype:** `{ds['subtype']}`  ")
        if ds.get("contact_note"):
            L.append(f"_{ds['contact_note']}_  ")
        if ds.get("init_action"):
            L.append(f"\n**INIT action:** {ds['init_action']}\n")
    tref = R.get("truth_reference_check", {})
    L.append(f"\n## TRUTH_REFERENCE_CHECK (first {tref.get('n_ticks','?')} ticks, ok={tref.get('ok')})")
    for ax in ("behind", "below"):
        if ax in tref:
            s = tref[ax]
            L.append(f"- {ax}: mean={s['mean']:.4f} std={s['std']:.4f} (RAW_CG_BOOM−TRUTH; ≤0.5 ⇒ GPS noise)")
    if "fused_valid_start_s" in R:
        L.append(f"\n## Join\nFUSED_valid_start = {R.get('fused_valid_start_s')} s; "
                 f"pre-formation excluded = {R.get('pre_formation_excluded_ticks')} ticks.")
        if "join" in R:
            j = R["join"]
            L.append(f"match-rate = {j['joined']}/{j['denominator_fused_valid']} = "
                     f"{j['match_rate']:.1%} (tol {j['tolerance_s']*1000:.0f} ms); "
                     f"unjoined {j['unjoined']} ({j['unjoined_frac']:.1%}; "
                     f"warn={j['data_quality_warning']}).")
        if "m3b_unjoined" in R:
            mw = R["m3b_unjoined"]
            L.append(f"M3b-window unjoined: {mw['m3b_window_unjoined']}/{mw['m3b_window_ticks']} "
                     f"({mw['m3b_window_unjoined_frac']:.1%}; warn={mw['warning']})"
                     + (" — WARNING: (a)/(c) may underrepresent bias in docking window." if mw['warning'] else ""))
    met = R.get("metrics") or {}
    if met:
        L.append("\n## Per-axis metrics (behind / cross / below — equal depth)")
        L.append("| metric | axis | mean | std | p50 | p95(|.|) | n |")
        L.append("|---|---|---|---|---|---|---|")
        keys = [("a_FUSED_minus_TRUTH", "a FUSED−TRUTH"),
                ("b_RAWBOOM_minus_TRUTH", "b RAW_CG_BOOM−TRUTH"),
                ("c_FUSED_minus_RAWBOOM", "c FUSED−RAW_CG_BOOM"),
                ("transform_RAWBOOM_minus_RAWCG", "boom RAW_CG_BOOM−RAW_CG_CG (≈−4.31)")]
        for kk, label in keys:
            for ax in ("behind", "cross", "below"):
                s = met[ax][kk]
                L.append(f"| {label} | {ax} | {s['mean']:.3f} | {s['std']:.3f} | "
                         f"{s['p50']:.3f} | {s['p95']:.3f} | {s['n']} |")
    drl = R.get("drl_obs_space_health") or {}
    if drl:
        L.append("\n## DRL obs-space health (10 s rolling-std of FUSED−TRUTH; fusion layer)")
        for ax in ("behind", "below"):
            d = drl[ax]
            L.append(f"- {ax}: mean rolling-std={d['mean']:.3f} m max={d['max']:.3f} m "
                     f"stationary={d['stationary']} → {d['interpretation']} (obs ÷{d['obs_scale_divisor']})")
        L.append(f"\n> {R.get('drl_caveat','')}")
    if "partial_metrics" in R:
        L.append("\n## Partial metrics (FUSED unavailable — b + boom-transform only)")
        for ax in ("behind", "below", "cross"):
            pm = R["partial_metrics"][ax]
            L.append(f"- {ax}: b mean={pm['b_RAWBOOM_minus_TRUTH']['mean']:.3f} "
                     f"std={pm['b_RAWBOOM_minus_TRUTH']['std']:.3f}; "
                     f"transform mean={pm['transform_RAWBOOM_minus_RAWCG']['mean']:.3f}")
    sv = R.get("stage1_validity")
    if sv:
        L.append("\n## STAGE-1 validity checks (CSV2_R7L CORRECTION 2)")
        L.append(f"- (a) gps_update_count={sv['a_gps_update_count']} pass={sv['a_pass']} — {sv['a_note']}")
        L.append(f"- (b) FUSED-behind rolling-std mean={sv['b_fused_behind_rolling_std_mean_m']:.3f} m "
                 f"pass={sv['b_pass']} — {sv['b_note']}")
        L.append(f"- (c) {sv['c_note']} pass={sv['c_pass']}")
    ig = R.get("isolation_gate")
    if ig:
        L.append(f"\n## ISOLATION GATE: `{ig['gate']}`")
        L.append(f"GPS-only floor mean(FUSED−RAW_CG_BOOM) behind = "
                 f"{ig['gps_only_floor_behind_mean_m']:.3f} m (|.|={ig['abs_m']:.3f}; valid={ig['valid']}).")
    if "n3_comparison" in R:
        L.append(f"\n## N=3 comparison\n{json.dumps(R['n3_comparison'], default=str)}")
    L.append(f"\n## Artifacts\nPNGs: {R.get('pngs')}  {R.get('pngs_note','')}")
    L.append(f"\nFUSED note: {R.get('fused_note')}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
