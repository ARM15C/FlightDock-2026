#!/usr/bin/env python3
# =============================================================================

#   the 2 x 2 of policy x 0.25 m exam-time offset at sigma 0.25 and sigma 0.35 -- from the board of record, on the
#   four-condition success flag (`passed`), with the exam's own flag beside it. PIDO amendment:
#   the grid is a 3 x 2 -- FlightDock-A2, FlightDock-S1 and the PID-baseline, each with and without the 0.25 m
#   down-offset -- so the offset's effect is measured on two learned policies AND on the baseline, and the learned
#   correction's effect can be read with the offset held fixed (S1O vs PIDO, A2O vs PIDO). Descriptive: no verdict.
#   Per cell: corridor p95 (legs that entered), attempted legs over the bar, overtake legs, legs that
#   never reached the corridor, safe-capture rate (nine stages) and strong-wind rate; paired within seed: the
#   offset's effect on success for each policy with its two-sided 95 % seed-cluster bootstrap range (Newcombe beside it); the sigma 0.25 -> 0.35
#   decrease per configuration; the paired change in held altitude. Descriptive: no verdict. "stage" never "rung".
# PUBLIC API: cell(board, tag) -> dict; effect(board, a, b, mask) -> dict; build(board) -> (DataFrame, dict);
#   main(argv=None) -> int. CLI: --run-dir DIR [--selftest]
# SIDE-EFFECTS: reads <run-dir>/board/BOARD_W10.csv; writes <run-dir>/verdicts/ABLATION_TABLE.csv, .md and
#   ABLATION_PLACEHOLDERS.json. Spawns nothing.
# SIZE: soft 110 / hard 140 code-LoC; function cap 40.
# COVERAGE: --selftest: a synthetic two-corpus board gives the known paired effect and held-altitude delta.
# =============================================================================
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hypothesis_verdicts import paired, newcombe_lb, BAR, ARM, NAME, ORDER, SHORT_SUF  # noqa: E402
import p95_boot as PB  # noqa: E402
import boot_rates as BR  # noqa: E402 (v4: the bootstrap is the range of record)


def tag(pre, sig):
    """Corpus tag: A2P025 / CSV025 / ARMB025; S1A25 / A2V25 / PIDO25 carry no leading zero."""
    return pre + (sig[1:] if pre in SHORT_SUF else sig)

CELLS = {k: (NAME[k], ARM[k]) for k in ORDER}   # key = shorthand, value = (name, corpus-tag prefix); order of record
#: (controller, no-offset corpus prefix, +0.25 m offset corpus prefix). PID/PIDO added 2026-09-17: with it the
#: 2 x 2 becomes a 3 x 2 and the offset's effect is read three times -- on two learned policies and on the baseline.
OFFSET_PAIRS = [("A2", "A2P", "A2V"), ("S1", "S1A", "CSV"), ("PID", "ARMB", "PIDO")]


def cell(board, tag, col="passed"):
    x = board[(board.corpus == tag) & board.wind_rung]
    ent = x[x.win.notna()]
    return {"corpus": tag, "n": len(x), "entered": len(ent), "never_reached": int(x.win.isna().sum()),
            "p95_corridor_m": round(PB.p95(ent.win), 3), "p95_bootstrap_ci95": PB.fmt(*PB.ci(x, "win")),
            "legs_over_bar": int((ent.win > BAR).sum()),
            "stages_over": int((ent.groupby("wind_label").win.apply(PB.p95) > BAR).sum()),
            "overtake_legs": int(x.real_overtake.astype(bool).sum()),
            "k_nine": int(x[col].sum()), "rate_nine_pct": round(100 * x[col].mean(), 1),
            "k_sw": int(x[x.sw][col].sum()), "rate_sw_pct": round(100 * x[x.sw][col].mean(), 1),
            "held_median_above_m": round(float(np.nanmedian(x.held_median_above_m)), 3) if x.held_median_above_m.notna().any() else np.nan}


def effect(board, a, b, mask, col="passed"):
    """Offset effect on success = (with offset) minus (without), paired within seed, with its two-sided 95 %
    seed-cluster bootstrap range (v4 range of record; the Wilson-based paired range is kept as a comparison); plus the paired change in held altitude."""
    r = paired(board, b, a, mask, col)
    nlo = newcombe_lb(r["a"], r["b"], r["c"], r["d"], 0.025)    # comparison
    nhi = -newcombe_lb(r["a"], r["c"], r["b"], r["d"], 0.025)
    lo, hi = BR.diff_ci(r)                                      # v4 range of record
    ha = board[(board.corpus == a) & mask(board)][["key", "held_median_above_m"]]
    hb = board[(board.corpus == b) & mask(board)][["key", "held_median_above_m"]]
    h = ha.merge(hb, on="key", suffixes=("_a", "_b")).dropna()
    dh = float((h.held_median_above_m_b - h.held_median_above_m_a).median()) if len(h) else np.nan
    return {"effect_pp": round(r["diff"], 2), "ci": "[%.1f, %.1f]" % (lo, hi), "ci_newcombe": "[%.1f, %.1f]" % (nlo, nhi),
            "disc": "%d / %d" % (r["b"], r["c"]),
            "held_delta_m": round(dh, 3) if dh == dh else "not computed", "n": r["n"]}


def p95_effect(board, a, b):
    """Offset effect on clearance = pooled p95 of the per-leg peak upward deviation (truth channel, inside the
    terminal-approach corridor, nine wind stages) WITH the offset minus WITHOUT, seed-cluster bootstrap interval."""
    x = board[board.wind_rung]
    pb, pa, d, lo, hi = PB.diff_ci(x[x.corpus == b], x[x.corpus == a], "win")
    return "%+.3f m %s" % (d, PB.fmt(lo, hi))


def build(board):
    rows, T = [], {}
    for key, (name, pre) in CELLS.items():
        for sig, s in (("025", "s025"), ("035", "s035")):
            c = cell(board, tag(pre, sig))
            ce = cell(board, tag(pre, sig), "passed_exam")
            c.update({"configuration": name, "sigma": "0." + sig[1:], "k_nine_exam": ce["k_nine"]})
            rows.append(c)
            T.update({"ABL.p95.%s.%s" % (key, s): "%.3f m %s" % (c["p95_corridor_m"], c["p95_bootstrap_ci95"]), "ABL.rate.%s.%s" % (key, s): "%d of 540 (%.1f %%)" % (c["k_nine"], c["rate_nine_pct"]),
                      "ABL.ratesw.%s.%s" % (key, s): "%d of 180 (%.1f %%)" % (c["k_sw"], c["rate_sw_pct"]), "ABL.legsover.%s.%s" % (key, s): c["legs_over_bar"]})
        d = paired(board, tag(pre, "025"), tag(pre, "035"), lambda x: x.wind_rung)
        T["ABL.decrease.%s" % key] = "%.1f pp" % d["diff"]
    for pol, a, b in OFFSET_PAIRS:
        for sig, s in (("025", "s025"), ("035", "s035")):
            e = effect(board, tag(a, sig), tag(b, sig), lambda x: x.wind_rung)
            esw = effect(board, tag(a, sig), tag(b, sig), lambda x: x.sw)
            T.update({"ABL.effect.%s.%s" % (pol, s): "%+.1f pp %s; strong-wind (6-8 m/s) %+.1f pp %s"
                      % (e["effect_pp"], e["ci"], esw["effect_pp"], esw["ci"]),
                      "ABL.p95effect.%s.%s" % (pol, s): p95_effect(board, tag(a, sig), tag(b, sig)),
                      "ABL.held.%s.%s" % (pol, s): ("%+.3f m" % e["held_delta_m"]) if isinstance(e["held_delta_m"], float) else e["held_delta_m"]})
    return pd.DataFrame(rows), T


def md_table(df):
    """Markdown table without the optional tabulate dependency."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def selftest():
    keys = ["W12|%d|s" % i for i in range(10)]
    mk = lambda tag, p, h: pd.DataFrame({"key": keys, "corpus": tag, "passed": p, "passed_exam": p, "sw": True, "wind_rung": True,
                                         "win": 1.0 + h, "seed": range(10), "real_overtake": False, "wind_label": "W12", "held_median_above_m": h})
    PB.BOOT = 50
    b = pd.concat([mk("S1A25", [True] * 5 + [False] * 5, -0.5), mk("CSV025", [True] * 7 + [False] * 3, -0.75),
                   mk("ARMB025", [True] * 4 + [False] * 6, -0.5), mk("PIDO25", [True] * 6 + [False] * 4, -0.75)])
    e = effect(b, "S1A25", "CSV025", lambda x: x.wind_rung)
    ok = abs(e["effect_pp"] - 20.0) < 1e-9 and abs(e["held_delta_m"] + 0.25) < 1e-9 and e["disc"] == "2 / 0" and e["ci"].startswith("[")
    p = effect(b, tag("ARMB", "025"), tag("PIDO", "025"), lambda x: x.wind_rung)   # the baseline's own offset pair
    ok = ok and abs(p["effect_pp"] - 20.0) < 1e-9 and len(OFFSET_PAIRS) == 3 and len(CELLS) == 6
    ok = ok and p95_effect(b, "S1A25", "CSV025") == "-0.250 m [-0.250, -0.250]"    # with the offset minus without
    print("SELFTEST", "PASS" if ok else "FAIL", e, p)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    board = pd.read_csv(os.path.join(a.run_dir, "board", "BOARD_W10.csv"), low_memory=False)
    for c in ("passed", "passed_exam", "sw", "wind_rung", "real_overtake"):
        board[c] = board[c].astype(str).str.lower().eq("true")
    tab, T = build(board)
    vd = os.path.join(a.run_dir, "verdicts")
    os.makedirs(vd, exist_ok=True)
    tab.to_csv(os.path.join(vd, "ABLATION_TABLE.csv"), index=False)
    open(os.path.join(vd, "ABLATION_TABLE.md"), "w").write("Channel: truth, inside the aerial-docking terminal approach corridor (<= 10.0 m of the boom tip); "
                                                            "success = the four-condition flag of record; k_nine_exam = the exam's whole-leg flag (reproduction).\n\n"
                                                            + md_table(tab) + "\n")
    json.dump(T, open(os.path.join(vd, "ABLATION_PLACEHOLDERS.json"), "w"), indent=1, default=str)
    print(tab[["corpus", "configuration", "sigma", "p95_corridor_m", "legs_over_bar", "k_nine", "k_nine_exam", "rate_sw_pct", "never_reached"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
