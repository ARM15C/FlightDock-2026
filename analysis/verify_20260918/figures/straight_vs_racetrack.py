#!/usr/bin/env python3
# =============================================================================

#   nothing. Sep "Straight vs. Racetrack" layout, sigma 0.25 panel stacked above sigma 0.35.
# WHAT IS DRAWN: docking-envelope captures (flag of record) pooled across the nine wind stages, per arm, on the straight
#   trajectory (solid bar) beside the racetrack trajectory (hatched, lighter bar); 270 legs per bar (30 seeds x 9 stages);
#   two-sided Wilson 95 % whiskers; count over each bar. Success has no truth / Perception version.
# SOURCES (nothing typed): board/BOARD_W10.csv; rates formatted by compare_arms.wilson.
# VALIDATION, ENFORCED (refuses): every cell (count, n, rate, interval) and the straight-minus-racetrack difference
#   equal SUCCESS_BY_TRAJECTORY.csv; 270 legs per bar; seed block 10775-10804.
# CLI: (none) SIDE-EFFECTS: writes STRAIGHT_VS_RACETRACK_POOLED_ACROSS_NINE_WIND_STAGES.png/.pdf/.csv/.md.
# =============================================================================
import os
import sys

import pandas as pd
from matplotlib.patches import Patch
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figure_style as S  # noqa: E402
import compare_arms as CA  # noqa: E402
from extra_analyses_ticklevel import NAMEARM  # noqa: E402

plt = S.plt
V = S.V
STEM = "STRAIGHT_VS_RACETRACK_POOLED_ACROSS_NINE_WIND_STAGES"
TRAJ = ("straight", "racetrack")
DESC = ("Straight vs. racetrack flight trajectories, pooled across the nine wind stages: docking-envelope captures (flag of record) as a share of legs flown,\n"
        "270 legs per bar (30 seeds x 9 stages); seed block 10775-10804. Whiskers: two-sided Wilson 95 % interval; count over each bar. Descriptive; not graded.")


def load():
    board = pd.read_csv(os.path.join(S.RUN, "board", "BOARD_W10.csv"), low_memory=False)
    for c in ("passed", "wind_rung"):
        board[c] = board[c].astype(str).str.lower().eq("true")
    ref = pd.read_csv(os.path.join(S.OUT, "SUCCESS_BY_TRAJECTORY.csv")).set_index(["arm", "sigma"])
    cells = {}
    for k in S.ARMS:
        for sig in (0.25, 0.35):
            x = board[(board.corpus == V.corpus(k, sig)) & board.wind_rung]
            if sorted(x.seed.unique()) != S.SEED_BLOCK:
                raise SystemExit("REFUSE: %s sigma %.2f is not seed block 10775-10804" % (k, sig))
            for tr in TRAJ:
                y = x[x.trajectory == tr]
                kk, n = int(y.passed.sum()), len(y)
                if n != 270 or "%d of %d, %s" % (kk, n, CA.wilson(kk, n)) != ref.loc[(NAMEARM[k], sig), tr]:
                    raise SystemExit("REFUSE: %s sigma %.2f %s does not reproduce SUCCESS_BY_TRAJECTORY.csv" % (k, sig, tr))
                lo, hi = proportion_confint(kk, n, alpha=0.05, method="wilson")
                cells[(k, sig, tr)] = {"k": kk, "n": n, "pct": 100.0 * kk / n, "lo": 100 * lo, "hi": 100 * hi}
            d = cells[(k, sig, "straight")]["pct"] - cells[(k, sig, "racetrack")]["pct"]
            if "%+.1f" % d != "%+.1f" % float(ref.loc[(NAMEARM[k], sig), "straight minus racetrack (pp)"]):
                raise SystemExit("REFUSE: %s sigma %.2f difference does not reproduce SUCCESS_BY_TRAJECTORY.csv" % (k, sig))
    print("VALIDATION PASS: 24 cells and 12 differences reproduce SUCCESS_BY_TRAJECTORY.csv; 270 legs per bar; seed block on 12 corpora")
    return cells


def figure(cells):
    fig, axs = plt.subplots(2, 1, figsize=(11, 9), sharex=True, gridspec_kw={"hspace": 0.07})
    for ax, sig in zip(axs, (0.25, 0.35)):
        for i, k in enumerate(S.ARMS):
            for j, tr in enumerate(TRAJ):
                c, x = cells[(k, sig, tr)], i + (j - 0.5) * 0.38
                ax.bar(x, c["pct"], width=0.36, facecolor=S.COLOUR[k] if tr == "straight" else S.COLOUR[k] + "66", edgecolor=S.DARK[k], lw=0.8,
                       hatch=None if tr == "straight" else "////", zorder=2)
                ax.errorbar(x, c["pct"], yerr=[[c["pct"] - c["lo"]], [c["hi"] - c["pct"]]], fmt="none", ecolor="k", elinewidth=0.9, capsize=3, zorder=3)
                ax.text(x, c["hi"] + 1.5, "%d/%d" % (c["k"], c["n"]), ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 10))
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("docking-envelope captures (% of legs flown)")
        ax.text(0.01, 0.97, "Sigma %.2f" % sig, transform=ax.transAxes, ha="left", va="top", fontsize=9.5)
    axs[1].set_xticks(range(len(S.ARMS)))
    axs[1].set_xticklabels([S.SHORT[k].replace(" (", "\n(") for k in S.ARMS])
    axs[0].set_title(DESC, fontsize=9, loc="left")
    h = [Patch(facecolor="0.45", edgecolor="0.2", label="straight (n = 270)"), Patch(facecolor="#bbbbbb66", edgecolor="0.2", hatch="////", label="racetrack (n = 270)")]
    h += [Patch(facecolor=S.COLOUR[k], edgecolor=S.DARK[k], label=S.SHORT[k]) for k in S.ARMS]
    fig.legend(handles=h, loc="upper center", ncol=4, framealpha=0.95, bbox_to_anchor=(0.5, 0.06))
    S.save(fig, STEM)


def table(cells):
    rows = []
    for sig in (0.25, 0.35):
        pid = cells[("PID", sig, "straight")]["pct"] - cells[("PID", sig, "racetrack")]["pct"]
        for k in S.ARMS:
            s, r = cells[(k, sig, "straight")], cells[(k, sig, "racetrack")]
            rows.append({"sigma": "%.2f" % sig, "arm": S.SHORT[k], "straight": "%d of %d, %s" % (s["k"], s["n"], CA.wilson(s["k"], s["n"])),
                         "racetrack": "%d of %d, %s" % (r["k"], r["n"], CA.wilson(r["k"], r["n"])),
                         "straight minus racetrack (pp)": "%+.1f" % (s["pct"] - r["pct"]),
                         "that difference minus the PID-baseline's (pp)": "%+.1f" % (s["pct"] - r["pct"] - pid)})
    t = pd.DataFrame(rows)
    stem = os.path.join(S.FIGDIR, STEM)
    t.to_csv(stem + ".csv", index=False)
    notes = ["Nine wind stages pooled, 270 legs per trajectory per arm and sigma (30 seeds x 9 stages), seed block 10775-10804. Success = the flag of record.",
             "Rates: two-sided Wilson 95 %. The differences are plain subtractions of rates (descriptive). Descriptive; not graded."]
    open(stem + ".md", "w").write("# Straight vs. racetrack flight trajectories, pooled across nine wind stages - POST-PLAN PRESENTATION; decides nothing\n\n"
                                  + S.md(t) + "\n\n" + "\n".join("- " + x for x in notes) + "\n")
    print("wrote", stem + ".csv|.md")


def main():
    cells = load()
    figure(cells)
    table(cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
