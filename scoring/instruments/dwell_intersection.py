#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: G1-v3 dwell instrumentation (Dispatch C, AUDIT, characterize
#   ONLY — never gates). Per leg: perception mask = ctrl_m3b flag, truth mask =
#   raw_m3b flag over the fixtured adjudicator's HC+postarrival rows; legacy
#   same-ts dwell (must reproduce archived adjudicator values); L-hat via
#   cross-correlation of ctrl_behind_m vs raw_behind_m (candidate procedure per
#   PREREG_G1V3.md; grid step = median d(sim_time_s), reported); v3 dwell =
#   longest P(t) AND T(t+L) run at L in {Lhat-1, Lhat, Lhat+1}; perception-only
#   and truth-only dwells; best v3 window bounds exported for the health pass.
# PUBLIC API: analyze_leg(leg_dir) -> dict; main(argv=None) -> int
#   (CLI: leg dirs..., --csv OUT, --fixture leg_dir EXPECT_DWELL).
# SIDE-EFFECTS: reads paired CSVs (read-only); writes --csv; prints to stdout.
#   No network, no process spawns, no SITL.
# CAPS: file soft 200 / hard 250 lines; functions <= 50 lines.
# COVERAGE: py_compile + PREREG fixture F1-F4 on #366 C1 (expect legacy 6.75 s)
#   BEFORE the 30-boot sweep.
# =============================================================================
import argparse
import csv
import importlib.util
import os
import sys

_ADJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                    "facetskip_20260709T205456Z", "facetskip_adjudicate.py")
_spec = importlib.util.spec_from_file_location("facetskip_adjudicate", _ADJ)
_adj = importlib.util.module_from_spec(_spec)
sys.modules["facetskip_adjudicate"] = _adj
_spec.loader.exec_module(_adj)

GAP = _adj.GAP           # 0.2 s gap tolerance — identical to every prior verdict
T = _adj.T               # "True"/"False" string coercion (parse-helpers class)
load = _adj.load         # HC-state + postarrival row filter (fixtured)


def mask_dwell(times, mask, gap=GAP):
    """Longest run (s) where mask is True, tolerating <=gap between hits."""
    best, rs, lt = 0.0, None, None
    for t, ok in zip(times, mask):
        if ok:
            if rs is None:
                rs = t
            elif lt is not None and t - lt > gap:
                best = max(best, lt - rs)
                rs = t
            lt = t
        else:
            if rs is not None and lt is not None:
                best = max(best, lt - rs)
            rs = lt = None
    if rs is not None and lt is not None:
        best = max(best, lt - rs)
    return best


def best_window(times, mask, gap=GAP):
    """(start,end) sim_time of the longest tolerated run (for the health pass)."""
    best, bi, rs, lt = 0.0, (None, None), None, None
    for t, ok in zip(times, mask):
        if ok:
            if rs is None:
                rs = t
            elif lt is not None and t - lt > gap:
                if lt - rs > best:
                    best, bi = lt - rs, (rs, lt)
                rs = t
            lt = t
        else:
            if rs is not None and lt is not None and lt - rs > best:
                best, bi = lt - rs, (rs, lt)
            rs = lt = None
    if rs is not None and lt is not None and lt - rs > best:
        best, bi = lt - rs, (rs, lt)
    return bi


def xcorr_lag(a, b, max_lag=10, margin=0.005):
    """argmax_k corr(diff a, diff b shifted k); prefer 0 unless the peak beats
    corr(0) by `margin`. INSTRUMENT AMENDMENT (disclosed, pre-sweep): level-
    series xcorr was weakly identified on #337-aligned channels (flat peak;
    fixture F4 caught a spurious lhat=-3) — first-differencing kills the drift
    and the parsimony margin prefers the aligned hypothesis.
    Positive k = b (truth) lags a (perception) by k steps."""
    da = [a[i + 1] - a[i] for i in range(len(a) - 1)]
    db = [b[i + 1] - b[i] for i in range(len(b) - 1)]
    n = len(da)
    if n < 2 * max_lag + 4:
        return 0
    def corr(k):
        pairs = [(da[i], db[i + k]) for i in range(max(0, -k), min(n, n - k))]
        va = sum(x * x for x, _ in pairs) or 1e-12
        vb = sum(y * y for _, y in pairs) or 1e-12
        return sum(x * y for x, y in pairs) / (va * vb) ** 0.5
    c0 = corr(0)
    best_k, best_c = 0, c0
    for k in range(-max_lag, max_lag + 1):
        c = corr(k)
        if c > best_c or (c == best_c and abs(k) < abs(best_k)):
            best_k, best_c = k, c
    return best_k if best_c - c0 > margin else 0


def shifted_and(p, t, k):
    """P(i) AND T(i+k) as a row-indexed mask (out-of-range -> False)."""
    n = len(p)
    return [p[i] and (0 <= i + k < n) and t[i + k] for i in range(n)]


def analyze_leg(leg_dir):
    rows = load(leg_dir)
    times = [float(r["sim_time_s"]) for r in rows]
    pmask = [T(r["ctrl_m3b"]) for r in rows]
    tmask = [T(r["raw_m3b"]) for r in rows]
    cb = [float(r["ctrl_behind_m"]) for r in rows]
    rb = [float(r["raw_behind_m"]) for r in rows]
    diffs = sorted(times[i + 1] - times[i] for i in range(len(times) - 1))
    step = diffs[len(diffs) // 2] if diffs else float("nan")
    lhat = xcorr_lag(cb, rb)
    legacy = mask_dwell(times, [p and t for p, t in zip(pmask, tmask)])
    v3 = {k: mask_dwell(times, shifted_and(pmask, tmask, k))
          for k in (lhat - 1, lhat, lhat + 1)}
    win = best_window(times, shifted_and(pmask, tmask, lhat))
    return {"leg": os.path.basename(leg_dir.rstrip("/")),
            "n_rows": len(rows), "grid_step_s": round(step, 4), "lhat": lhat,
            "legacy_dwell": round(legacy, 3),
            "v3_lm1": round(v3[lhat - 1], 3), "v3_l": round(v3[lhat], 3),
            "v3_lp1": round(v3[lhat + 1], 3),
            "p_only": round(mask_dwell(times, pmask), 3),
            "t_only": round(mask_dwell(times, tmask), 3),
            "win_start": win[0], "win_end": win[1]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="G1-v3 dwell intersection")
    ap.add_argument("leg_dirs", nargs="*")
    ap.add_argument("--csv")
    ap.add_argument("--fixture", nargs=2, metavar=("LEG", "EXPECT"),
                    help="fixture leg dir + expected legacy dwell (F1-F4 gate)")
    args = ap.parse_args(argv)
    if args.fixture:
        r = analyze_leg(args.fixture[0])
        exp = float(args.fixture[1])
        f1 = abs(r["legacy_dwell"] - exp) <= 0.01
        f2 = r["p_only"] >= r["legacy_dwell"] and r["t_only"] >= r["legacy_dwell"]
        f3 = abs(r["lhat"]) <= 3
        f4 = r["v3_l"] >= r["legacy_dwell"] - 0.01
        print(f"FIXTURE {r['leg']}: legacy={r['legacy_dwell']} (expect {exp}) "
              f"lhat={r['lhat']} v3={r['v3_l']} p_only={r['p_only']} "
              f"t_only={r['t_only']} | F1={f1} F2={f2} F3={f3} F4={f4}")
        return 0 if all((f1, f2, f3, f4)) else 2
    out = [analyze_leg(d) for d in args.leg_dirs]
    cols = list(out[0].keys())
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(out)
    for r in out:
        print(f"{r['leg']:>4} | step={r['grid_step_s']:.3f}s lhat={r['lhat']:+d}"
              f" | legacy={r['legacy_dwell']:5.2f} v3(L-1/L/L+1)="
              f"{r['v3_lm1']:5.2f}/{r['v3_l']:5.2f}/{r['v3_lp1']:5.2f}"
              f" | P-only={r['p_only']:5.2f} T-only={r['t_only']:5.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
