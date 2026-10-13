#!/usr/bin/env python3
# =============================================================================

#   decide nothing. One pass over all 7,200 flight_state tick files. Two channels only: truth =
#   raw_*, perception = ctrl_*. Success = the flag of record (board `passed`).
#   Definitions COPIED from the Sep outputs that carried them (FlightDock_figures/paper/tables/*.csv and
#   the September table set); box tolerances IMPORTED from src/drl/config.py (CONTACT_M3B_*_TOL), never typed.
#   Row 15 APPROACH FUNNEL (Fig4-6 stages): flown; within 10 m of the tip; within 3 m; within 1.5 m; inside the M3b box;
#     in the box >= 1 s cumulative; >= 3 s cumulative (ticks x tick period, not necessarily continuous); M3b success.
#     Each geometric step on truth AND on perception. Pools: all ten stages, and the nine wind stages.
#   Row 18 ENVELOPE CAPTURE ALONG THE APPROACH (Fig4-8, "at each fore-and-aft distance from the boom-tip"): for d in
#     10, 5, 3, 1.5 m, at the FIRST tick of the leg with |behind| <= d, the share of legs whose cross and below errors
#     are already inside tolerance (each, and both); then the share that reached the box on all three axes at least
#     once, and M3b success. Nine wind stages; denominator = all legs flown. Truth and perception.
#   Row 17 ENVELOPE GEOMETRY (Fig4-7): per axis |behind|, |cross|, |below| over every corridor tick (truth range <=
#     10.0 m): p5, p25, p50, p75, p95 and the share inside tolerance; truth and perception; delta = perception minus truth.
#   Row 4 PERCEPTION MINUS TRUTH, VERTICAL, BY STAGE: perception height minus truth height (height = -below) per
#     corridor tick: p5, p50, p95, per corpus and stage (the strip under the Sep the figure “H1: p95 Peak Upward Deviation, Light Turbulence”).
#   Row 24 HELD POSITION INSIDE THE BOX, BY STAGE: ticks inside the box on all three axes (truth): held
#     height median on truth and perception, delta; ticks held below the tip (%); applied altitude residual median; ticks
#     asking UP on the altitude axis (%).
#   Row 25 TIME BELOW THE BOX: every corridor tick in one of four truth states - below the box (below >
#     tol), inside on all three axes, in the vertical band but off-axis, above the box (below < -tol). Per leg share
#     below; paired with the PID-baseline on (stage, seed, trajectory); median difference; seed-paired Wilcoxon
#     signed-rank (scipy.stats.wilcoxon). Of below-box ticks, COPIED from the Sep code
#     (analysis/ch4pull_20260904T1605Z/ch4_below.py `_resid`, ch4_below_render.py `_resid`): share asking up
#     (res_policy_2 > 0); of those, kept (res_applied_2 > 0), suppressed (== 0), flipped down (< 0); median applied
#     altitude residual while below the box; each a per-leg value, reported as the MEDIAN OVER LEGS.
#     Nine wind stages. Perception share below given beside truth.
#   VALIDATION (refuses): every file found (7,200); the tick period is one value; wind-stage corridor tick counts equal
#     step 23's per-leg counts; every successful leg has >= 3 s cumulative in-box time on truth.
# CLI: --extract [--workers 12] | --report SIDE-EFFECTS: writes EXTRA_TL_*.npz/.csv (intermediates) and the named
#   report files under OUT only.
# =============================================================================
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import concurrent.futures as cf
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from offset_delivery import OUT, RUN, TAGS, arm, md  # noqa: E402
from src.drl import config as C  # noqa: E402

TOL = {"behind": C.CONTACT_M3B_BEHIND_TOL, "cross": C.CONTACT_M3B_CROSS_TOL, "below": C.CONTACT_M3B_BELOW_TOL}
VIEW = {"truth": "raw", "perception": "ctrl"}
DS = (10.0, 5.0, 3.0, 1.5)
COLS = ["sim_time_s", "res_policy_2", "res_applied_2"] + ["%s_%s_m" % (v, a) for v in VIEW.values() for a in ("range", "behind", "cross", "below")]
NAMEARM = {"A2": "FlightDock-A2", "A2O": "FlightDock-A2O", "S1": "FlightDock-S1", "S1O": "FlightDock-S1O", "PID": "PID-baseline (comparator)", "PIDO": "PIDO"}


def inbox(d, v):
    return ((d["%s_behind_m" % v].abs() <= TOL["behind"]) & (d["%s_cross_m" % v].abs() <= TOL["cross"])
            & (d["%s_below_m" % v].abs() <= TOL["below"])).to_numpy()


def capture(d, v):
    out = {}
    b = d["%s_behind_m" % v].abs().to_numpy()
    cin = (d["%s_cross_m" % v].abs() <= TOL["cross"]).to_numpy()
    bin_ = (d["%s_below_m" % v].abs() <= TOL["below"]).to_numpy()
    for dist in DS:
        idx = np.flatnonzero(b <= dist)
        out["%s_d%g_reached" % (v, dist)] = len(idx) > 0
        i = idx[0] if len(idx) else None
        for nm, m in (("cross", cin), ("below", bin_), ("both", cin & bin_)):
            out["%s_d%g_%s" % (v, dist, nm)] = bool(m[i]) if i is not None else False
    return out


def one(job):
    tag, path = job
    leg = path.split(os.sep)[-3]
    d = pd.read_csv(path, compression="gzip", usecols=COLS, low_memory=False)
    dt = float(np.median(np.diff(d.sim_time_s.dropna().to_numpy()))) if d.sim_time_s.notna().sum() > 1 else np.nan
    row = {"tag": tag, "leg": leg, "ticks": len(d), "dt": dt}
    for name, v in VIEW.items():
        r = d["%s_range_m" % v]
        ib = inbox(d, v)
        row.update({"%s_min_range" % v: r.min(), "%s_w10" % v: bool((r <= 10.0).any()), "%s_w3" % v: bool((r <= 3.0).any()),
                    "%s_w15" % v: bool((r <= 1.5).any()), "%s_inbox_ticks" % v: int(ib.sum())})
        row.update(capture(d, v))
    corr = (d.raw_range_m <= 10.0).to_numpy()
    k = d[corr]
    row["corridor_ticks"] = len(k)
    below_t, below_f = k.raw_below_m.to_numpy(), k.ctrl_below_m.to_numpy()
    ib_t = inbox(k, "raw")
    st_below, st_above = below_t > TOL["below"], below_t < -TOL["below"]
    row.update({"st_below": int(st_below.sum()), "st_in": int(ib_t.sum()), "st_above": int(st_above.sum()),
                "st_band_offaxis": int((~st_below & ~st_above & ~ib_t).sum()), "perception_below": int((below_f > TOL["below"]).sum())})
    pol, app = k.res_policy_2.to_numpy(), k.res_applied_2.to_numpy()
    up, nb = st_below & (pol > 0), int(st_below.sum())
    nu = int(up.sum())
    row.update({"below_share_asking_up": nu / nb if nb else np.nan,
                "up_share_kept": float((app[up] > 0).mean()) if nu else np.nan,
                "up_share_suppressed": float((app[up] == 0).mean()) if nu else np.nan,
                "up_share_flipped_down": float((app[up] < 0).mean()) if nu else np.nan,
                "below_applied_median": float(np.nanmedian(app[st_below])) if nb else np.nan})
    geo = np.column_stack([k.raw_behind_m, k.raw_cross_m, k.raw_below_m, k.ctrl_behind_m, k.ctrl_cross_m, k.ctrl_below_m]).astype(np.float32)
    held = np.column_stack([below_t[ib_t], below_f[ib_t], app[ib_t], pol[ib_t]]).astype(np.float32)
    return row, geo, held


def extract(workers):
    jobs = [(t, p) for t in TAGS for p in sorted(glob.glob(os.path.join(RUN, "shims", t, "flat", "*", "env*", "flight_state_ep*.csv.gz")))]
    if len(jobs) != 7200:
        raise SystemExit("REFUSE: expected 7200 flight_state files, found %d" % len(jobs))
    rows = []
    for t in TAGS:
        geo, held = {}, {}
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            for row, g, h in ex.map(one, [j for j in jobs if j[0] == t], chunksize=10):
                rows.append(row)
                st = row["leg"].split("_")[0]
                geo.setdefault(st, []).append(g)
                held.setdefault(st, []).append(h)
        np.savez(os.path.join(OUT, "EXTRA_TL_TICKS_%s.npz" % t), **{"geo_" + s: np.concatenate(v) for s, v in geo.items()},
                 **{"held_" + s: np.concatenate(v) for s, v in held.items()})
        print("%s legs=%d" % (t, sum(1 for r in rows if r["tag"] == t)), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "EXTRA_TL_PER_LEG.csv"), index=False)
    print("EXTRACT DONE legs=%d" % len(rows))


def load():
    L = pd.read_csv(os.path.join(OUT, "EXTRA_TL_PER_LEG.csv"), low_memory=False)
    if "fused_below" in L.columns:   # extract of 2026-09-18 used the old name
        L = L.rename(columns={"fused_below": "perception_below"})
        L.to_csv(os.path.join(OUT, "EXTRA_TL_PER_LEG.csv"), index=False)
    b = pd.read_csv(os.path.join(RUN, "board", "BOARD_W10.csv"), low_memory=False, usecols=["corpus", "leg", "key", "seed", "trajectory", "wind_label", "wind_rung", "passed"])
    b["passed"], b["wind_rung"] = b.passed.astype(str).eq("True"), b.wind_rung.astype(str).eq("True")
    L = L.merge(b, left_on=["tag", "leg"], right_on=["corpus", "leg"], how="inner")
    if len(L) != 7200:
        raise SystemExit("REFUSE: per-leg join has %d rows, expected 7200" % len(L))
    dts = L.dt.round(4).unique()
    if len(dts) != 1:
        raise SystemExit("REFUSE: more than one tick period: %s" % dts[:5])
    R = pd.read_csv(os.path.join(OUT, "RESIDUAL_STATISTICS_TICK_SOURCE_PER_LEG.csv"))
    m = L[L.wind_rung].merge(R[["tag", "leg", "corridor_ticks"]], on=["tag", "leg"], suffixes=("", "_step23"))
    if len(m) != 6480 or (m.corridor_ticks != m.corridor_ticks_step23).any():
        raise SystemExit("REFUSE: wind-stage corridor tick counts do not equal step 23's (%d legs joined)" % len(m))
    dt = float(dts[0])
    bad = L[L.passed & (L.raw_inbox_ticks * dt < 3.0 - 1e-9)]
    if len(bad):
        raise SystemExit("REFUSE: %d successful legs have < 3 s cumulative in-box time on truth" % len(bad))
    print("VALIDATION: 7,200 legs joined; tick period %.4f s; 6,480 wind-stage corridor counts equal step 23; every success has >= 3 s in the box on truth" % dt)
    return L, dt


def wil(k, n):
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return "%d of %d, %.1f %% [%.1f, %.1f]" % (k, n, 100.0 * k / n, 100 * lo, 100 * hi)


def sig(t):
    return "0.25" if t.endswith("25") else "0.35"


def stage_name(s):
    return "calm control stage" if s == "S0" else "%d m/s" % (int(s[1:]) // 2)


def funnel(L, dt):
    rows = []
    for t in TAGS:
        for pool, x in (("all ten stages", L[L.tag == t]), ("nine wind stages", L[(L.tag == t) & L.wind_rung])):
            n = len(x)
            steps = [("flown", n, n)]
            for lab, key in (("within 10 m of the tip", "w10"), ("within 3 m", "w3"), ("within 1.5 m", "w15")):
                steps.append((lab, int(x["raw_" + key].sum()), int(x["ctrl_" + key].sum())))
            for lab, sec in (("inside the M3b box", 0.0), ("in the box >= 1 s cumulative", 1.0), ("in the box >= 3 s cumulative", 3.0)):
                f = (lambda c: int((x[c] > 0).sum())) if sec == 0 else (lambda c, s=sec: int((x[c] * dt >= s - 1e-9).sum()))
                steps.append((lab, f("raw_inbox_ticks"), f("ctrl_inbox_ticks")))
            steps.append(("M3b success (flag of record)", int(x.passed.sum()), None))
            for lab, kt, kf in steps:
                rows.append({"arm": NAMEARM[arm(t)], "sigma": sig(t), "pool": pool, "step": lab, "truth": wil(kt, n),
                             "perception": "" if kf is None else wil(kf, n), "perception minus truth (legs)": "" if kf is None else "%+d" % (kf - kt)})
    return pd.DataFrame(rows)


def capture_table(L):
    rows = []
    for t in TAGS:
        x = L[(L.tag == t) & L.wind_rung]
        n = len(x)
        for dist in DS:
            for nm, lab in (("cross", "cross-track inside tolerance"), ("below", "vertical inside tolerance"), ("both", "both inside tolerance")):
                kt, kf = int(x["raw_d%g_%s" % (dist, nm)].sum()), int(x["ctrl_d%g_%s" % (dist, nm)].sum())
                rows.append({"arm": NAMEARM[arm(t)], "sigma": sig(t), "first tick with |behind| <= (m)": "%g" % dist, "quantity": lab,
                             "truth": wil(kt, n), "perception": wil(kf, n), "perception minus truth (legs)": "%+d" % (kf - kt)})
        kt, kf = int((x.raw_inbox_ticks > 0).sum()), int((x.ctrl_inbox_ticks > 0).sum())
        rows.append({"arm": NAMEARM[arm(t)], "sigma": sig(t), "first tick with |behind| <= (m)": "any", "quantity": "reached the box on all three axes at least once",
                     "truth": wil(kt, n), "perception": wil(kf, n), "perception minus truth (legs)": "%+d" % (kf - kt)})
        rows.append({"arm": NAMEARM[arm(t)], "sigma": sig(t), "first tick with |behind| <= (m)": "any", "quantity": "M3b success (flag of record)",
                     "truth": wil(int(x.passed.sum()), n), "perception": "", "perception minus truth (legs)": ""})
    return pd.DataFrame(rows)


def q(a):
    return np.percentile(a, [5, 25, 50, 75, 95]) if len(a) else np.full(5, np.nan)


def geometry_and_strip():
    G, S = [], []
    for t in TAGS:
        z = np.load(os.path.join(OUT, "EXTRA_TL_TICKS_%s.npz" % t))
        stages = sorted([k[4:] for k in z.files if k.startswith("geo_")], key=lambda s: (s != "S0", int(s[1:]) if s != "S0" else -1))
        wind = np.concatenate([z["geo_" + s] for s in stages if s != "S0"])
        for pool, a in [("nine wind stages", wind)] + [(stage_name(s), z["geo_" + s]) for s in stages]:
            for i, ax in enumerate(("behind", "cross", "below")):
                tq, fq = q(np.abs(a[:, i])), q(np.abs(a[:, 3 + i]))
                G.append({"arm": NAMEARM[arm(t)], "sigma": sig(t), "pool": pool, "axis |m|": ax, "tolerance (m)": TOL[ax], "corridor ticks": len(a),
                          "truth p5 / p25 / p50 / p75 / p95": " / ".join("%.3f" % v for v in tq),
                          "perception p5 / p25 / p50 / p75 / p95": " / ".join("%.3f" % v for v in fq),
                          "median, perception minus truth": "%+.3f" % (fq[2] - tq[2]),
                          "inside tolerance, truth (%)": "%.1f" % (100 * (np.abs(a[:, i]) <= TOL[ax]).mean()),
                          "inside tolerance, perception (%)": "%.1f" % (100 * (np.abs(a[:, 3 + i]) <= TOL[ax]).mean())})
            gap = (-a[:, 5]) - (-a[:, 2])
            p5, p50, p95 = np.percentile(gap, [5, 50, 95]) if len(gap) else (np.nan,) * 3
            S.append({"arm": NAMEARM[arm(t)], "sigma": sig(t), "pool": pool, "corridor ticks": len(a),
                      "truth height p50 (m)": "%.3f" % np.median(-a[:, 2]), "perception height p50 (m)": "%.3f" % np.median(-a[:, 5]),
                      "perception minus truth, height: p5 / p50 / p95 (m)": "%+.3f / %+.3f / %+.3f" % (p5, p50, p95)})
    return pd.DataFrame(G), pd.DataFrame(S)


def held_by_stage():
    rows = []
    for t in TAGS:
        z = np.load(os.path.join(OUT, "EXTRA_TL_TICKS_%s.npz" % t))
        stages = sorted([k[5:] for k in z.files if k.startswith("held_")], key=lambda s: (s != "S0", int(s[1:]) if s != "S0" else -1))
        wind = np.concatenate([z["held_" + s] for s in stages if s != "S0"])
        for pool, h in [(stage_name(s), z["held_" + s]) for s in stages] + [("nine wind stages", wind)]:
            ht, hf = -h[:, 0], -h[:, 1]
            rows.append({"arm": NAMEARM[arm(t)], "sigma": sig(t), "stage": pool, "ticks inside the box (truth)": len(h),
                         "held height above the tip, truth: median (m)": "%+.3f" % np.median(ht) if len(h) else "",
                         "perception: median (m)": "%+.3f" % np.median(hf) if len(h) else "",
                         "perception minus truth (m)": "%+.3f" % (np.median(hf) - np.median(ht)) if len(h) else "",
                         "ticks held below the tip, truth (%)": "%.1f" % (100 * (ht < 0).mean()) if len(h) else "",
                         "applied altitude correction: median": "%+.3f" % np.nanmedian(h[:, 2]) if len(h) else "",
                         "ticks asking UP on altitude (%)": "%.1f" % (100 * (h[:, 3] > 0).mean()) if len(h) else ""})
    return pd.DataFrame(rows)


def below_box(L):
    W = L[L.wind_rung & (L.corridor_ticks > 0)].copy()
    W["share_below"], W["share_below_perception"] = W.st_below / W.corridor_ticks, W.perception_below / W.corridor_ticks
    pid = {"0.25": "ARMB025", "0.35": "ARMB035"}
    rows = []
    for t in TAGS:
        x = W[W.tag == t]
        p = W[W.tag == pid[sig(t)]][["key", "share_below"]]
        m = x.merge(p, on="key", suffixes=("", "_pid"))
        d = m.share_below - m.share_below_pid
        is_pid = t == pid[sig(t)]
        w = None if is_pid or (d != 0).sum() == 0 else wilcoxon(m.share_below, m.share_below_pid)
        pw = "" if w is None else ("%.2g%s" % (w.pvalue, " *" if w.pvalue < 0.05 else ""))
        learned = arm(t) in ("A2", "A2O", "S1", "S1O")
        bt = int(x.st_below.sum())
        rows.append({"arm": NAMEARM[arm(t)], "sigma": sig(t), "legs with corridor ticks": len(x), "paired legs with the PID-baseline": "" if is_pid else len(m),
                     "share of corridor ticks below the box, truth: median": "%.3f" % x.share_below.median(),
                     "perception: median": "%.3f" % x.share_below_perception.median(),
                     "PID-baseline on the same legs: median": "" if is_pid else "%.3f" % m.share_below_pid.median(),
                     "paired median difference": "" if is_pid else "%+.3f" % d.median(),
                     "seed-paired Wilcoxon signed-rank p": pw,
                     "below-box ticks (truth)": bt,
                     "of those, asking up: median over legs (%)": "%.1f" % (100.0 * x.below_share_asking_up.median()) if learned else "",
                     "of upward asks, kept (applied > 0): median over legs (%)": "%.1f" % (100.0 * x.up_share_kept.median()) if learned else "",
                     "suppressed (= 0) (%)": "%.1f" % (100.0 * x.up_share_suppressed.median()) if learned else "",
                     "flipped down (< 0) (%)": "%.1f" % (100.0 * x.up_share_flipped_down.median()) if learned else "",
                     "applied altitude correction while below the box: median over legs": "%+.3f" % x.below_applied_median.median() if learned else ""})
    return pd.DataFrame(rows)


NOTES = {
    "APPROACH_FUNNEL": "Each step counts legs that reached it (k of n, share, two-sided Wilson 95 %%). Truth = raw_*, perception = ctrl_* (what the controller reads); box = |behind| <= %.1f, |cross| <= %.1f, |below| <= %.1f m (config CONTACT_M3B_*_TOL). Cumulative time = ticks inside the box x the tick period, not necessarily continuous. Success = the flag of record (3.0 s continuous)." % (TOL["behind"], TOL["cross"], TOL["below"]),
    "ENVELOPE_CAPTURE_ALONG_THE_APPROACH": "Nine wind stages, 540 legs per corpus, denominator = all legs flown. At the FIRST tick of the leg with |behind| <= d on that view: are the cross-track and vertical errors already inside tolerance? Two-sided Wilson 95 %.",
    "ENVELOPE_GEOMETRY_IN_CORRIDOR": "Every tick with truth range <= 10.0 m (perception values read on the SAME ticks). |value| quantiles over ticks, not legs.",
    "PERCEPTION_MINUS_TRUTH_VERTICAL_BY_STAGE": "Every tick with truth range <= 10.0 m. Height = -below (m above the boom-tip). Positive = perception reads the QuadPlane higher than truth.",
    "HELD_POSITION_IN_BOX_BY_STAGE": "Ticks inside the box on all three axes on truth, truth range <= 10.0 m. Height above the tip is negative below the tip. Applied correction = res_applied_2 (includes the gated 0.25 m offset in A2O, S1O and PIDO). PID-baseline applied = 0 by construction.",
    "TIME_BELOW_THE_BOX": "Nine wind stages. Four truth states per corridor tick: below the box, inside, vertical band but off-axis, above. Pairing with the PID-baseline on (stage, seed, trajectory). * = Wilcoxon p < 0.05. 'Asking up' columns only for the four FlightDock arms (no learned correction acts in PID-baseline or PIDO)."}


def report():
    L, dt = load()
    out = {"APPROACH_FUNNEL": funnel(L, dt), "ENVELOPE_CAPTURE_ALONG_THE_APPROACH": capture_table(L)}
    out["ENVELOPE_GEOMETRY_IN_CORRIDOR"], out["PERCEPTION_MINUS_TRUTH_VERTICAL_BY_STAGE"] = geometry_and_strip()
    out["HELD_POSITION_IN_BOX_BY_STAGE"], out["TIME_BELOW_THE_BOX"] = held_by_stage(), below_box(L)
    for stem, t in out.items():
        t.to_csv(os.path.join(OUT, stem + ".csv"), index=False)
        open(os.path.join(OUT, stem + ".md"), "w").write("# %s - POST-PLAN; decides nothing\n\n" % stem.replace("_", " ").lower().capitalize()
                                                         + md(t) + "\n\n- " + NOTES[stem] + "\n")
        print("wrote %s (%d rows)" % (stem, len(t)))


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
