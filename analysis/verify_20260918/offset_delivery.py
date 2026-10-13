#!/usr/bin/env python3
# =============================================================================

# QUESTION: (1) where is the fixed 0.25 m downward altitude offset actually applied, and is it on in the second
#   before an altitude-guard fire; (2) while it is commanded, is it delivered, or taken back?
# MECHANISM READ FROM THE CODE (src/drl/env_formation_sitl.py 2858-2868, 2908-2923; residual_action.py 264-283;
#   config MILESTONE_THRESHOLDS[2] = 1.5/1.5/1.5 m, BIAS_SMOOTH_ALPHA 0.15): each tick the offset target is 0.25 m
#   when the milestone tier is >= 2 (aircraft within 1.5 m behind / cross-track / below of its CURRENT target) and
#   0 otherwise, passed through an EMA; the logged res_applied_2 is (final altitude command - baseline command).
# MEASURED OFFSET PER TICK: PID-baseline / PIDO have zero learned correction, so voff = -res_applied_2 exactly.
#   For a policy arm voff_est = res_policy_2 - res_applied_2; this is exact only while the policy's scaled output
#   equals its raw output, so the two NO-offset policy arms (true voff = 0) are carried as the control that prices
#   the estimate's error, and `clean` = share of corridor ticks with voff_est inside [-0.005, 0.255].
# CHANNEL: truth (raw_*) and Perception (ctrl_*, what the controller reads) side by side.
#   Corridor = ticks with raw_range_m <= 10.0. Height above the boom-tip = -raw_below_m (truth), -ctrl_below_m (Perception).
# CLI: --extract [--workers 12] (reads every leg's flight_state READ-ONLY via the run's flat shims)
#      --report (tables from the two extract CSVs)
# SIDE-EFFECTS: writes OFFSET_LEGS.csv, OFFSET_PREFIRE.csv, OFFSET_REPORT.md under --out only.
# =============================================================================
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import concurrent.futures as cf
import glob
import os
import sys

import numpy as np
import pandas as pd

RUN = (_REPO + "/results/run_20260918T2208Z")
OUT = (_REPO + "/results/derived")
TAGS = ["A2P025", "A2P035", "A2V25", "A2V35", "S1A25", "S1A35", "CSV025", "CSV035", "ARMB025", "ARMB035",
        "PIDO25", "PIDO35"]
NAME = {"A2P": "A2", "A2V": "A2O", "S1A": "S1", "CSV": "S1O", "ARMB": "PID", "PIDO": "PIDO"}
PAIRS = [("A2V", "A2P"), ("CSV", "S1A"), ("PIDO", "ARMB")]
COLS = ["sim_time_s", "raw_below_m", "ctrl_below_m", "raw_range_m", "res_policy_2", "res_applied_2",
        "_guard_altitude_ceiling_fire_count"]
BANDS = [(-np.inf, 0.0), (0.0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, np.inf)]
ON, OFF, BOOT, SEED = 0.2375, 0.0125, 10000, 20260915          # on = >= 95 % of 0.25 m; off = <= 5 %


def arm(tag):
    return NAME[tag.rstrip("0123456789")]


def one(job):
    tag, leg, path = job
    d = pd.read_csv(path, compression="gzip", usecols=COLS, low_memory=False)
    zero_policy = arm(tag) in ("PID", "PIDO")
    d["voff"] = -d.res_applied_2 if zero_policy else d.res_policy_2 - d.res_applied_2
    d["up"], d["up_p"] = -d.raw_below_m, -d.ctrl_below_m
    c = d[d.raw_range_m <= 10.0]
    row = {"tag": tag, "leg": leg, "n_ticks": len(d), "n_corr": len(c)}
    if len(c):
        row.update(up_mean=c.up.mean(), up_median=c.up.median(), up_p_mean=c.up_p.mean(), applied_mean=c.res_applied_2.mean(),
                   policy_mean=c.res_policy_2.mean(), voff_mean=c.voff.mean(), n_on=int((c.voff >= ON).sum()),
                   n_off=int((c.voff <= OFF).sum()), n_clean=int(c.voff.between(-0.005, 0.255).sum()))
        for i, (lo, hi) in enumerate(BANDS):
            b = c[(c.up > lo) & (c.up <= hi)]
            row["band%d_n" % i], row["band%d_on" % i] = len(b), int((b.voff >= ON).sum())
            bp = c[(c.up_p > lo) & (c.up_p <= hi)]
            row["pband%d_n" % i], row["pband%d_on" % i] = len(bp), int((bp.voff >= ON).sum())
    fires = []
    cnt = d["_guard_altitude_ceiling_fire_count"].fillna(0).to_numpy()
    for i in np.flatnonzero(np.diff(cnt) > 0) + 1:
        t = d.sim_time_s.iloc[i]
        last = d[(d.sim_time_s >= t - 1.0) & (d.sim_time_s <= t)]
        early = d[(d.sim_time_s >= t - 3.0) & (d.sim_time_s <= t - 2.0)]
        fires.append({"tag": tag, "leg": leg, "t_fire": t, "up_at_fire": d.up.iloc[i], "range_at_fire": d.raw_range_m.iloc[i],
                      "voff_at_fire": d.voff.iloc[i], "voff_last_1s": last.voff.median(),
                      "voff_3_to_2s_before": early.voff.median() if len(early) else np.nan,
                      "up_3_to_2s_before": early.up.median() if len(early) else np.nan,
                      "up_p_at_fire": d.up_p.iloc[i], "up_p_3_to_2s_before": early.up_p.median() if len(early) else np.nan})
    return row, fires


def extract(workers):
    jobs = []
    for tag in TAGS:
        for p in sorted(glob.glob(os.path.join(RUN, "shims", tag, "flat", "*", "env*", "flight_state_ep*.csv.gz"))):
            jobs.append((tag, p.split(os.sep)[-3], p))
    per = pd.Series([j[0] for j in jobs]).value_counts().to_dict()
    if any(per.get(t) != 600 for t in TAGS):
        raise SystemExit("REFUSE: expected 600 flight_state files per corpus, found %s" % per)
    rows, fires = [], []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (r, f) in enumerate(ex.map(one, jobs, chunksize=20)):
            rows.append(r)
            fires += f
            if k % 600 == 599:
                print("legs read:", k + 1, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "OFFSET_LEGS.csv"), index=False)
    pd.DataFrame(fires).to_csv(os.path.join(OUT, "OFFSET_PREFIRE.csv"), index=False)
    print("EXTRACT DONE legs=%d fires=%d" % (len(rows), len(fires)))


def boot_mean(x, seeds):
    """Mean of x with a two-sided 95 % seed-cluster bootstrap interval (whole seeds resampled)."""
    g = pd.DataFrame({"x": x, "s": seeds}).groupby("s").x.agg(["sum", "size"]).to_numpy()
    pick = np.random.default_rng(SEED).integers(0, len(g), size=(BOOT, len(g)))
    est = g[pick, 0].sum(1) / g[pick, 1].sum(1)
    return float(np.mean(x)), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def md(df):
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
                     + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def bands_table(W, tags, fmt):
    """Share of ticks with the offset fully on, per height band, on truth and on Perception (one row per channel)."""
    rows = []
    for ch, pre in (("truth", "band"), ("perception", "pband")):
        b = W.groupby("tag")[["%s%d_%s" % (pre, i, k) for i in range(len(BANDS)) for k in ("n", "on")]].sum().reindex(tags)
        r = pd.DataFrame({"tag": tags, "arm": [arm(t) for t in tags], "height channel": ch})
        for i, (lo, hi) in enumerate(BANDS):
            lab = "<= 0 m" if i == 0 else "> 2.0 m" if i == 4 else "%.1f-%.1f m" % (lo, hi)
            r[fmt % lab] = (100 * b["%s%d_on" % (pre, i)] / b["%s%d_n" % (pre, i)]).round(1).to_numpy()
        rows.append(r)
    order = {t: i for i, t in enumerate(tags)}
    out = pd.concat(rows, ignore_index=True)
    out["_o"] = out.tag.map(order) * 2 + (out["height channel"] == "perception")   # truth row first, then perception
    return out.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def report():
    L = pd.read_csv(os.path.join(OUT, "OFFSET_LEGS.csv"))
    F = pd.read_csv(os.path.join(OUT, "OFFSET_PREFIRE.csv"))
    L["wind"] = ~L.leg.str.startswith("S0_")
    L["seed"] = L.leg.str.split("_").str[1].astype(int)
    W = L[L.wind & (L.n_corr > 0)]
    a = W.groupby("tag").agg(legs=("leg", "size"), ticks=("n_corr", "sum"), on=("n_on", "sum"), off=("n_off", "sum"),
                             clean=("n_clean", "sum")).reindex(TAGS).reset_index()
    a["arm"] = a.tag.map(arm)
    for c in ("on", "off", "clean"):
        a[c + "_pct"] = (100 * a[c] / a.ticks).round(1)
    a["partial_pct"] = (100 - a.on_pct - a.off_pct).round(1)
    tm = W.assign(w=W.voff_mean * W.n_corr).groupby("tag").apply(lambda g: g.w.sum() / g.n_corr.sum(), include_groups=False)
    a["mean_offset_m"] = a.tag.map(tm).round(3)
    A = a[["tag", "arm", "legs", "ticks", "on_pct", "partial_pct", "off_pct", "mean_offset_m", "clean_pct"]]
    off = [t for t in TAGS if arm(t) == "PIDO"]      # the only offset arm whose per-tick offset is exact (control, table A)
    B = bands_table(W[W.tag.isin(off)], off, "on %% at height %s")
    Fw = F[~F.leg.str.startswith("S0_") & (F.range_at_fire <= 10.0)]
    C = Fw.groupby("tag").agg(fires=("leg", "size"), up_at_fire_median=("up_at_fire", "median"),
                              offset_3_to_2s_before=("voff_3_to_2s_before", "median"),
                              up_3_to_2s_before=("up_3_to_2s_before", "median"),
                              up_p_at_fire_median=("up_p_at_fire", "median"), up_p_3_to_2s_before=("up_p_3_to_2s_before", "median"),
                              offset_last_1s=("voff_last_1s", "median"), offset_at_fire=("voff_at_fire", "median"),
                              fires_offset_off_at_fire_pct=("voff_at_fire", lambda s: 100 * (s <= OFF).mean())
                              ).reindex([t for t in TAGS if arm(t) in ("PID", "PIDO")]).round(3).reset_index()
    C.insert(1, "arm", C.tag.map(arm))
    C["at fire, perception minus truth (m)"] = (C.up_p_at_fire_median - C.up_at_fire_median).round(3)
    C = C.rename(columns={"up_at_fire_median": "height at fire, truth (m)", "up_p_at_fire_median": "height at fire, perception (m)",
                          "up_3_to_2s_before": "3-2 s before, truth (m)", "up_p_3_to_2s_before": "3-2 s before, perception (m)"})
    rows = []
    for wo, no in PAIRS:
        for sg in ("25", "35"):
            tw = [t for t in TAGS if t.startswith(wo) and t.endswith(sg)][0]
            tn = [t for t in TAGS if t.startswith(no) and t.endswith(sg)][0]
            m = W[W.tag == tw].merge(W[W.tag == tn], on=["leg", "seed"], suffixes=("_w", "_n"))
            dh = boot_mean((m.up_mean_w - m.up_mean_n).to_numpy(), m.seed.to_numpy())
            dhp = boot_mean((m.up_p_mean_w - m.up_p_mean_n).to_numpy(), m.seed.to_numpy())
            cmd = boot_mean((-m.voff_mean_w).to_numpy(), m.seed.to_numpy())
            dap = boot_mean((m.applied_mean_w - m.applied_mean_n).to_numpy(), m.seed.to_numpy())
            f = lambda v: "%+.3f [%+.3f, %+.3f]" % v
            exact = arm(tw) == "PIDO"
            rows.append({"controller": "%s -> %s" % (arm(tn), arm(tw)), "sigma": "0." + sg, "paired legs": len(m),
                         "offset commanded, mean (m)": f(cmd) if exact else "not separable",
                         "height change delivered, truth, mean (m)": f(dh),
                         "height change delivered, perception, mean (m)": f(dhp),
                         "perception minus truth (m)": "%+.3f" % (dhp[0] - dh[0]),
                         "delivered / commanded": "%.2f" % (dh[0] / cmd[0]) if exact else "not separable",
                         "change in total applied altitude term (m)": f(dap)})
    D = pd.DataFrame(rows)
    txt = ["# Offset delivery - POST-PLAN DIAGNOSTIC, not in the signed plans; decides nothing", "",
           "Truth and Perception side by side; nine wind stages; corridor = ticks within 10.0 m of the boom-tip; legs that entered it.",
           "Offset 'on' = >= %.4f m (95 %% of 0.25); 'off' = <= %.4f m. Intervals: seed-cluster bootstrap, %d resamples,"
           " seed %d." % (ON, OFF, BOOT, SEED), "",
           "## A. Where the offset is applied (share of corridor ticks)", "",
           "CONTROL: A2, S1 and PID carry NO offset, so their 'on' share must be 0. PID reads 0 and PIDO is 100 % clean,",
           "so PID/PIDO are exact. The no-offset POLICY arms read far from 0: a policy's raw output and applied value",
           "already differ (bound, floor), so the per-tick offset is NOT separable for A2O / S1O from these logs.",
           "Tables B and C therefore cover PIDO only; table D gives the policy arms' delivered change without a split.", "",
           md(A), "", "## B. Offset fully on, by height above the boom-tip on truth and on Perception (PIDO)", "", md(B), "",
           "## C. Around each altitude-guard fire inside the corridor (medians over fires)", "", md(C), "",
           "## D. Commanded against delivered (paired within seed, stage and trajectory; with minus without)", "",
           "delivered / commanded near 1 = the aircraft goes where the offset sends it; well under 1 = taken back.", "",
           md(D), ""]
    open(os.path.join(OUT, "OFFSET_REPORT.md"), "w").write("\n".join(txt))
    pd.set_option("display.width", 250)
    for name, t in (("A", A), ("B", B), ("C", C), ("D", D)):
        print("\n== %s ==" % name)
        print(t.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    if a.extract:
        extract(a.workers)
    if a.report:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
