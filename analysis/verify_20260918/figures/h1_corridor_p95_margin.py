#!/usr/bin/env python3
# =============================================================================

#   POST-PLAN PRESENTATION; decides nothing. Verdict words follow the table “The Five FSM States That Decide an Aerial-Docking Leg” (v3) and PVALUE_BOOTSTRAP_20260922.csv
#   : H1 is MET when the
#   upper end of the two-sided 95% seed-cluster bootstrap interval of the corridor p95 of the per-leg peak upward deviation
#   (truth channel, sigma 0.25, nine wind stages pooled, legs that entered the terminal approach corridor <= 10 m) is below 2.00 m.
# WHAT IS DRAWN: a COLUMN chart at the portrait print width. Six columns (S1O, S1, A2O, A2, PIDO,
#   PID), height = truth p95 of the per-leg peak upward deviation in meters above the boom tip, LOWER IS BETTER. Column fill
#   = the arm's shade of record (dark = with the 0.25 m offset, light = without), dark edge; value and interval in white inside the column, centered on the 1.0 m level so every column's text is aligned. On each column top a BLACK capped
#   whisker = the two-sided 95% seed-cluster bootstrap interval (as on the chart). Dotted orange 2.00 m = guard
#   trip / H1 bar; solid red 3.42 m = DHC-2 airframe contact plane. Under each column: value, interval, n, verdict (green "H1
#   met"). Truth only (perception is on the chart). Crop note under the chart.
# SOURCES (nothing typed): per-leg truth corridor peaks = board `win` (BOARD_W10.csv via amplifying_tables.arm_sigma, wind-stage
#   legs); p95 and interval RECOMPUTED here with hypothesis_alternatives_v4.pct_ci (p95_boot picks: 30 seeds whole, 10,000
#   resamples, seed 20260915).
# VALIDATION, ENFORCED (refuses): recomputed p95 and interval == CORRIDOR_DIST_TRUTH_AND_PERCEPTION.csv (3 dp) == HYPOTHESIS_
#   VERDICTS.csv H1 row == PVALUE_BOOTSTRAP_20260922.csv H1 rows (value, ci95_lo, ci95_hi, bar 2.00) and its A_interval_clears
#   flag; seed block 10775-10804; 540 wind-stage legs per arm.
# CLI: (none) SIDE-EFFECTS: writes H1_CORRIDOR_P95_MARGIN.png/.pdf/.csv/.md under OUT/figures.
# =============================================================================
# 2026-10-01 ADVISOR RULING: POOLED WINDS ONLY. The strong-wind pool, strong_wind_pools_appendix.py and their
#   outputs are ARCHIVED in archive_strong_wind_20261001/; older comments below that mention the figure are historical.
#   WORDING: "comparator (not graded)" replaces "reported, not graded" for the PID-baseline and PIDO; the PID H2 row reads
#   "baseline comparator (not graded)"; sigma 0.35 and other ungraded readings say "not graded for H1".
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import os
import re
import sys

import numpy as np
import pandas as pd
from matplotlib.markers import MarkerStyle
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figure_style as S  # noqa: E402
sys.path.insert(0, (_REPO + "/scoring/tools"))
import amplifying_tables as AT  # noqa: E402
sys.path.insert(0, (_REPO + "/analysis/whatif_20260920"))
import hypothesis_alternatives_v4 as HA  # noqa: E402

plt = S.plt
V = S.V
STEM = "H1_CORRIDOR_P95_MARGIN"
ROWS = list(S.ROW_ORDER)
SIG = 0.25
BAR = float(S.GUARD_TRIP)
NUM = re.compile(r"[-+]?\d+\.?\d*")
PVAL = (_REPO + "/analysis/pvalue_bootstrap_20260922/PVALUE_BOOTSTRAP_20260922.csv")
PV_ARM = {"PID": "PID-baseline"}
TITLE = "H1 Margin: Corridor p95 of the Peak Upward Deviation by Arm, Truth Channel, Light Turbulence (σ = 0.25)"
LABEL = "Nine wind stages pooled (0 to 8 m/s, σ=0.25), truth channel, legs that entered the terminal approach corridor: the H1 test"
DESC = ("The 95th percentile (p95) of the per-leg peak upward deviation above the boom-tip inside the terminal approach corridor (within 10 m "
        "of the boom-tip), truth channel, nine wind stages pooled under light turbulence (σ = 0.25), 540 legs flown per arm, seed block "
        "10775-10804; n = legs that entered the corridor. Column height = the p95 in meters; the black capped whisker on each column is its "
        "two-sided 95% seed-cluster bootstrap interval (30 seeds resampled whole, 10,000 draws). The dotted orange line at 2.00 m is the guard "
        "trip and the H1 bar; the solid red line at 3.42 m is the DHC-2 airframe contact plane; the shaded band between them is the 1.42 m built-in margin, the room the guard has to arrest a climb and break away; lower is better. H1 is tested on the objective "
        "simulation truth data: it is met when the whole interval lies below 2.00 m. The PID-baseline and PIDO are comparators (not graded). Dark "
        "columns are the arms with the 0.25 m offset (S1O, A2O, PIDO); light columns are the same controllers without it (S1, A2, PID). The "
        "per-leg distributions, the perception channel and σ = 0.35 are in the document.")
FIGURE_NOTE = ("Column = corridor p95 of the per-leg peak upward deviation, truth channel; dark shade = with the 0.25 m offset (S1O, A2O, PIDO), "
               "light shade = without (S1, A2, PID). Black capped whisker = two-sided 95% seed-cluster bootstrap interval. Dotted orange line = 2.00 m, "
               "the guard trip and the H1 bar; solid red line = 3.42 m, the DHC-2 airframe contact plane; shaded band between them = the 1.42 m built-in margin for the guard's arrest and breakaway. Lower is better.")
PERC_MARKER = MarkerStyle("d").rotated(deg=90)   # sideways thin diamond
NOTE_IN = 1.25


def nums(cell):
    return tuple(float(x) for x in NUM.findall(str(cell)))


def load():
    board = pd.read_csv(os.path.join(S.RUN, "board", "BOARD_W10.csv"), low_memory=False)
    b = AT.arm_sigma(board).merge(pd.read_csv(os.path.join(S.OUT, "PERCEPTION_PEAKS_PER_LEG.csv")), on=["corpus", "leg"], how="left", validate="one_to_one")
    if not ((b.win.isna() == b.truth_corridor.isna()).all() and float((b.win - b.truth_corridor).abs().max()) <= 1e-4):
        raise SystemExit("REFUSE: step-A truth corridor peak does not equal the board's win")
    b = b[b.wind_rung.astype(str).str.lower().eq("true")]
    comp = pd.read_csv(os.path.join(S.OUT, "CORRIDOR_DIST_TRUTH_AND_PERCEPTION.csv"))
    comp = comp[comp.stage == "nine wind stages pooled"]
    h1 = pd.read_csv(os.path.join(S.RUN, "verdicts", "HYPOTHESIS_VERDICTS.csv")).set_index("hypothesis").loc["H1"]
    pv = pd.read_csv(PVAL)
    pv = pv[pv.hypothesis == "H1"].set_index("arm")
    cells, words = {}, {}
    for k in ROWS:
        x = b[b.corpus == V.corpus(k, SIG)]
        if len(x) != 540 or sorted(x.seed.unique()) != S.SEED_BLOCK:
            raise SystemExit("REFUSE: %s has %d wind-stage legs or a wrong seed block" % (k, len(x)))
        p, lo, hi = HA.pct_ci(x, [95])
        p, lo, hi = round(float(p[0]), 3), round(float(lo[0]), 3), round(float(hi[0]), 3)
        r = comp[(comp.arm == V.NAME[k]) & (comp.sig == SIG)]
        if len(r) != 1 or (p,) + nums(r["p95 bootstrap 95 %, truth"].iloc[0]) != (p, lo, hi) or float(r["p95, truth"].iloc[0]) != p:
            raise SystemExit("REFUSE: %s recomputed p95 / interval does not reproduce CORRIDOR_DIST_TRUTH_AND_PERCEPTION.csv" % k)
        if nums(h1[V.NAME[k]]) != (p, lo, hi):
            raise SystemExit("REFUSE: %s does not reproduce the HYPOTHESIS_VERDICTS.csv H1 row" % k)
        q = pv.loc[PV_ARM.get(k, k)]
        if (round(float(q.value), 3), round(float(q.ci95_lo), 3), round(float(q.ci95_hi), 3), float(q.bar)) != (p, lo, hi, BAR):
            raise SystemExit("REFUSE: %s does not reproduce PVALUE_BOOTSTRAP_20260922.csv" % k)
        met = hi < BAR
        if bool(q.A_interval_clears) != met:
            raise SystemExit("REFUSE: %s bound-rule verdict differs from PVALUE_BOOTSTRAP A_interval_clears" % k)
        words[k] = ("H1 met" if met else "H1 not met") if k in V.POL else "comparator (not graded)"
        xp = x[["seed", "perception_corridor"]].rename(columns={"perception_corridor": "win"})
        pp, plo, phi = HA.pct_ci(xp, [95])
        pp, plo, phi = round(float(pp[0]), 3), round(float(plo[0]), 3), round(float(phi[0]), 3)
        if (pp,) + nums(r["p95 bootstrap 95 %, perception"].iloc[0]) != (pp, plo, phi) or float(r["p95, perception"].iloc[0]) != pp:
            raise SystemExit("REFUSE: %s recomputed PERCEPTION p95 / interval does not reproduce CORRIDOR_DIST_TRUTH_AND_PERCEPTION.csv" % k)
        cells[k] = dict(p95=p, two=(lo, hi), perc=pp, perc_two=(plo, phi), n=int(x.win.notna().sum()), p_one=float(q.boot_p_one_sided))
    print("VALIDATION PASS: 6 recomputed truth AND perception p95 + intervals reproduce CORRIDOR_DIST, HYPOTHESIS_VERDICTS and PVALUE_BOOTSTRAP_20260922 (A_interval_clears); words %s" % words)
    return cells, words


def draw_panel(fig, ax, cells, words):
    """Column chart: height = truth p95, black capped whisker = its interval, rule lines at 2.00 m and 3.42 m, verdicts under the ticks."""
    for i, k in enumerate(ROWS):
        c = cells[k]
        ax.bar(i, c["p95"], width=0.62, facecolor=S.COLOUR[k], edgecolor=S.DARK[k], linewidth=1.4, zorder=3)
        ax.text(i, 1.0, "%.3f m\n[%.3f, %.3f]" % (c["p95"], c["two"][0], c["two"][1]), ha="center", va="center", color="white", fontsize=S.FONT_ROW - 0.5,
                fontweight="bold", linespacing=1.2, zorder=6, path_effects=[pe.withStroke(linewidth=1.2, foreground=S.DARK[k])])
        ax.errorbar([i], [c["p95"]], yerr=[[c["p95"] - c["two"][0]], [c["two"][1] - c["p95"]]], fmt="none", ecolor="k", elinewidth=1.2, capsize=4, capthick=1.2, zorder=5)
    ax.axhline(BAR, color=S.ORANGE, lw=2.0, ls=(0, (2, 2)), zorder=4)
    ax.axhline(S.CONTACT, color=S.RED, lw=1.8, zorder=4)
    # the 1.42 m built-in margin between the bar and the contact plane: shaded band + double arrow at the right
    ax.axhspan(BAR, S.CONTACT, facecolor=S.ORANGE, alpha=0.07, lw=0, zorder=0)
    xa = len(ROWS) - 0.4 + 0.12
    ax.annotate("", xy=(xa, S.CONTACT), xytext=(xa, BAR), arrowprops=dict(arrowstyle="<->", color=S.ORANGE, lw=1.2, shrinkA=0, shrinkB=0), zorder=6)
    ax.text(xa - 0.07, (BAR + S.CONTACT) / 2, "%.2f m built-in margin:\nroom for the guard to arrest\na climb and break away" % (S.CONTACT - BAR), color=S.ORANGE, ha="right", va="center", fontsize=S.FONT_ROW, zorder=6)
    ax.text(-0.58, BAR + 0.03, "2.00 m: H1 bar", color=S.ORANGE, ha="left", va="bottom", fontweight="bold", fontsize=S.FONT_ROW, zorder=6)
    ax.text(len(ROWS) - 0.5, S.CONTACT + 0.03, "3.42 m: DHC-2 airframe contact plane", color=S.RED, ha="right", va="bottom", fontweight="bold", fontsize=S.FONT_ROW, zorder=6)
    ax.set_xticks(range(len(ROWS)))
    ax.set_xticklabels(["%s\nn = %d" % (S.SHORT[k], cells[k]["n"]) for k in ROWS], fontsize=S.FONT_ROW, linespacing=1.25)
    ax.set_xlim(-0.6, len(ROWS) - 0.4 + 0.22)
    ax.set_ylim(0, S.CONTACT + 0.3)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("95th-percentile peak height above the boom-tip (m)\nlower is better")   # package 10: plain label; no in-figure title (the caption carries it)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for t in ax.get_xticklabels():
        if t.get_window_extent(r).width > ax.get_window_extent(r).width / len(ROWS) - 2:
            raise SystemExit("REFUSE: an x tick label is wider than its slot")
    for i, k in enumerate(ROWS):
        w = words[k]
        ax.annotate(w.replace(" (", "\n("), xy=(i, 0), xycoords=("data", "axes fraction"), xytext=(0, -22 * S.FONT_ROW / 5.0), textcoords="offset points", ha="center", va="top",
                    fontsize=S.FONT_ROW, color=S.GREEN if w == "H1 met" else "k", fontweight="bold" if w == "H1 met" else "normal")


def figure(cells, words):
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(S.TEXT_W_PORTRAIT, 3.2 + NOTE_IN), gridspec_kw={"height_ratios": [3.2, NOTE_IN]})
    fig.set_layout_engine("constrained")
    draw_panel(fig, ax, cells, words)
    S.crop_note(fig, axn, "Title: " + TITLE + " || Description: " + DESC + " || Legend: " + FIGURE_NOTE, S.TEXT_W_PORTRAIT)
    S.save_exact(fig, STEM, S.TEXT_W_PORTRAIT)


def table(cells, words):
    rows = [{"arm": S.SHORT[k], "H1 verdict (bound rule, truth, sigma 0.25)": words[k], "legs flown": 540, "legs that entered the corridor (n)": c["n"],
             "corridor p95, truth (m)": "%.3f" % c["p95"], "two-sided 95% seed-cluster bootstrap interval, truth (m)": "[%.3f, %.3f]" % c["two"],
             "corridor p95, perception (m), not graded": "%.3f" % c["perc"], "two-sided 95% seed-cluster bootstrap interval, perception (m)": "[%.3f, %.3f]" % c["perc_two"],
             "one-sided bootstrap p (H0: p95 >= 2.00 m)": "%.4f" % c["p_one"]} for k, c in ((k, cells[k]) for k in ROWS)]
    t = pd.DataFrame(rows)
    stem = os.path.join(S.FIGDIR, STEM)
    t.to_csv(stem + ".csv", index=False)
    notes = ["H1 statistic: corridor p95 of the per-leg peak upward deviation above the boom-tip, truth channel, inside the terminal approach corridor (within 10 m), nine wind stages pooled at sigma 0.25; 540 legs flown per arm, seed block 10775-10804; legs that never entered the corridor have no corridor peak.",
             "H1 rule (the table “The Five FSM States That Decide an Aerial-Docking Leg”): met when the upper end of the two-sided 95% seed-cluster bootstrap interval lies below 2.00 m (one-sided test at alpha 0.025). Whole seeds are resampled, 10,000 draws, fixed random seed; the 2.5th and 97.5th percentiles bound the interval. Verdict words from PVALUE_BOOTSTRAP_20260922.csv.",
             "H1 is tested on the objective simulation truth channel. Only the FlightDock controllers are graded; the PID-baseline and PIDO receive no verdict. Perception, sigma 0.35 and the per-leg distributions are in the figure (its p95 is kept in this table for reference).",
             "Column chart: height is the p95 in meters; lower is better. Rule lines: 2.00 m guard trip / H1 bar; 3.42 m DHC-2 airframe contact plane.",
             "Palette: " + S.PALETTE_RULE + "."]
    open(stem + ".md", "w").write("# H1 corridor p95 margin - POST-PLAN PRESENTATION; decides nothing\n\nSuggested figure title: " + TITLE +
                                  "\n\nFigure description (for the document, not drawn in the figure): " + DESC + "\n\nFigure note (the legend, for your 9 pt note under the figure): " + FIGURE_NOTE + "\n\n"
                                  + S.md(t) + "\n\n" + "\n".join("- " + x for x in notes) + "\n")
    print("wrote", stem + ".csv|.md")


def main():
    cells, words = load()
    figure(cells, words)
    table(cells, words)
    return 0


if __name__ == "__main__":
    sys.exit(main())
