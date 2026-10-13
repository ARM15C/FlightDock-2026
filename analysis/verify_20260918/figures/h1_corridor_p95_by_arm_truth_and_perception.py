#!/usr/bin/env python3
# =============================================================================

#   : the chapter's H1 verdict chart is now h1_corridor_p95_margin.py (row layout). This figure keeps the
#   per-leg DISTRIBUTIONS: box = per-leg truth corridor peaks of the legs that entered the corridor (median, IQR, whiskers
#   p5 to p95 so the top whisker IS the p95, dots beyond); thick black line = truth p95 with its seed-cluster bootstrap
#   interval; hollow circle further right = the Perception p95 and its interval (not graded). Left group sigma 0.25 (H1
#   test), right group sigma 0.35 (comparator (not graded)). Axis clipped at -1 m; legs below counted in grey. Dotted orange =
#   guard trip / H1 bar 2.00 m; solid red = airframe contact plane 3.42 m. POST-PLAN PRESENTATION; decides nothing.
# 2026-10-01 RULES: landscape print width 9.0 in, 7 pt, palette of record (arm order S1O, S1, A2O, A2, PIDO, PID; dark =
#   with the offset), verdict words from PVALUE_BOOTSTRAP_20260922.csv (bound rule; "H1 met" in green), legend row, crop note.
# SOURCES (nothing typed): per-leg peaks PERCEPTION_PEAKS_PER_LEG.csv (truth == board `win`); p95 and intervals
#   CORRIDOR_DIST_TRUTH_AND_PERCEPTION.csv; HYPOTHESIS_VERDICTS.csv; PVALUE_BOOTSTRAP_20260922.csv.
# VALIDATION, ENFORCED (refuses): the box sample's p95 equals the tabled truth p95 (3 dp) for every arm and sigma; every sigma
#   0.25 truth p95 and interval equals the H1 row of HYPOTHESIS_VERDICTS.csv and the PVALUE_BOOTSTRAP H1 rows with their
#   A_interval_clears flag; seed block 10775-10804; 540 wind-stage legs per arm and sigma; no Perception interval above the axis.
# CLI: (none) SIDE-EFFECTS: writes H1_CORRIDOR_P95_BY_ARM_TRUTH_AND_PERCEPTION.png/.pdf/.csv/.md under OUT/figures.
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.markers import MarkerStyle
from matplotlib.legend_handler import HandlerErrorbar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figure_style as S  # noqa: E402
sys.path.insert(0, (_REPO + "/scoring/tools"))
import amplifying_tables as AT  # noqa: E402

plt = S.plt
STEM = "H1_CORRIDOR_P95_BY_ARM_TRUTH_AND_PERCEPTION"
NUM = re.compile(r"[-+]?\d+\.?\d*")
YLIM, CLIP = (-1.6, 4.0), -1.0                          # Sep layout; top 4.0
SIGS, GAP = (0.25, 0.35), 0.8
ROWS = list(S.ROW_ORDER)
PVAL = (_REPO + "/analysis/pvalue_bootstrap_20260922/PVALUE_BOOTSTRAP_20260922.csv")
PV_ARM = {"PID": "PID-baseline"}
NOTE_IN, LEG_IN = 1.0, (1.1 if S.SLIDE else 0.75)
PERC_MARKER = MarkerStyle("d").rotated(deg=90)
LABEL = "Nine wind stages pooled (0 to 8 m/s), per-leg corridor peaks by arm, truth boxes with the perception p95 beside each; σ 0.25 is the H1 test (truth channel), σ 0.35 is not graded for H1"
if S.SLIDE:
    LABEL = "Nine wind stages pooled (0 to 8 m/s): per-leg corridor peaks by arm, truth boxes with the perception p95 beside each"
TITLE = "Per-Leg Peak Upward Deviation in the Terminal Approach Corridor by Arm, Truth and Perception, σ = 0.25 and σ = 0.35"
DESC_NOTE = ("Per-leg peak upward deviation above the boom-tip inside the terminal approach corridor (within 10 m), nine wind stages pooled, "
             "540 legs flown per arm and σ, seed block 10775-10804; n = legs that entered the corridor (the rest never came within 10 m and have "
             "no corridor peak). Boxes: truth, median and IQR; whiskers p5 to p95, so the top whisker is the H1 statistic; dots = legs beyond the "
             "whiskers; the axis is clipped at -1 m and legs below are counted in gray. The thick black line is the truth p95 with its two-sided "
             "95% seed-cluster bootstrap interval; the sideways diamond is the perception p95 (what the controller read, same legs) with its "
             "interval, not graded. Left group: σ = 0.25, the H1 test; H1 is met when the truth interval lies below 2.00 m (verdict under each "
             "box). Right group: σ = 0.35, not graded for H1. Dotted orange = 2.00 m guard trip and H1 bar; solid red = 3.42 m airframe "
             "contact plane; the shaded band between them is the 1.42 m built-in margin, the room the guard has to arrest a climb and break away. Dark shades are the arms with the 0.25 m offset (S1O, A2O, PIDO); light shades the same controllers without it.")


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
    cells, n_legs = {}, set()
    for k in S.ARMS:
        for sig in SIGS:
            x = b[b.corpus == S.V.corpus(k, sig)]
            r = comp[(comp.arm == S.V.NAME[k]) & (comp.sig == sig)]
            if len(r) != 1 or sorted(x.seed.unique()) != S.SEED_BLOCK:
                raise SystemExit("REFUSE: %s sigma %.2f: %d pooled rows, seeds %s" % (k, sig, len(r), sorted(x.seed.unique())[:3]))
            r = r.iloc[0]
            t = (float(r["p95, truth"]),) + nums(r["p95 bootstrap 95 %, truth"])
            p = (float(r["p95, perception"]),) + nums(r["p95 bootstrap 95 %, perception"])
            peaks = x.truth_corridor.dropna().to_numpy()
            if round(float(np.percentile(peaks, 95)), 3) != t[0]:
                raise SystemExit("REFUSE: box sample p95 of %s sigma %.2f does not equal the tabled p95" % (k, sig))
            if sig == 0.25 and nums(h1[S.V.NAME[k]]) != t:
                raise SystemExit("REFUSE: %s truth p95 differs from HYPOTHESIS_VERDICTS.csv" % k)
            if p[2] > YLIM[1]:
                raise SystemExit("REFUSE: a Perception interval runs above the axis")
            if round(float(box_stats(peaks)["whishi"]), 3) != t[0]:
                raise SystemExit("REFUSE: top whisker of %s sigma %.2f is not the tabled p95" % (k, sig))
            n_legs.add(len(x))
            cells[(k, sig)] = {"peaks": peaks, "truth": t, "perception": p, "perc_minus_truth": r["p95, perception minus truth"]}
    if n_legs != {540}:
        raise SystemExit("REFUSE: wind-stage legs per arm and sigma are %s, expected 540" % sorted(n_legs))
    pv = pd.read_csv(PVAL)
    pv = pv[pv.hypothesis == "H1"].set_index("arm")
    words = {}
    for k in S.ARMS:   # bound rule, from the bootstrap file
        q = pv.loc[PV_ARM.get(k, k)]
        t = cells[(k, 0.25)]["truth"]
        if (round(float(q.value), 3), round(float(q.ci95_lo), 3), round(float(q.ci95_hi), 3)) != t or float(q.bar) != S.GUARD_TRIP:
            raise SystemExit("REFUSE: %s does not reproduce PVALUE_BOOTSTRAP_20260922.csv" % k)
        met = t[2] < S.GUARD_TRIP
        if bool(q.A_interval_clears) != met:
            raise SystemExit("REFUSE: %s bound-rule verdict differs from A_interval_clears" % k)
        words[k] = ("H1 met" if met else "H1 not met") if k in S.V.POL else "comparator (not graded)"
    print("VALIDATION PASS: box p95 == tabled p95 on 12 cells; sigma 0.25 == HYPOTHESIS_VERDICTS.csv; seed block and 540 legs on 12 corpora; verdicts %s" % words)
    return cells, words


def box_stats(v):
    """Whiskers p5 to p95; legs beyond them are dots."""
    lo, q1, med, q3, hi = np.percentile(v, [5, 25, 50, 75, 95])
    return {"med": med, "q1": q1, "q3": q3, "whislo": lo, "whishi": hi, "fliers": v[((v < lo) | (v > hi)) & (v >= CLIP)]}   # legs below the clip are counted, not drawn


def figure(cells, words):
    fig = plt.figure(figsize=(S.TEXT_W_LANDSCAPE, 4.2 + LEG_IN + NOTE_IN))
    gs = fig.add_gridspec(3, 1, height_ratios=[4.2, LEG_IN, NOTE_IN])
    ax, axl, axn = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    axl.set_axis_off()
    fig.set_layout_engine("constrained")
    pos, labels, colours = [], [], []
    for j, sig in enumerate(SIGS):
        for i, k in enumerate(ROWS):
            xp = j * (len(ROWS) + GAP) + i
            c = cells[(k, sig)]
            st = box_stats(c["peaks"])
            bp = ax.bxp([st], positions=[xp], widths=0.5, patch_artist=True, showfliers=True,
                        flierprops=dict(marker="o", ms=2.2, mfc=S.COLOUR[k], mec=S.DARK[k], mew=0.4, alpha=0.85))
            bp["boxes"][0].set_facecolor(S.COLOUR[k]); bp["boxes"][0].set_edgecolor(S.DARK[k]); bp["medians"][0].set_color("k")
            for part in ("whiskers", "caps"):
                for a in bp[part]:
                    a.set_color(S.DARK[k])
            (t, tlo, thi), (p, plo, phi) = c["truth"], c["perception"]
            ax.plot([xp - 0.25, xp + 0.25], [t, t], color="k", lw=2.0, zorder=5)
            ax.errorbar([xp + 0.33], [t], yerr=[[t - tlo], [thi - t]], fmt="_", color="k", ms=5, capsize=2, elinewidth=0.9, zorder=5)
            ax.errorbar([xp + 0.45], [p], yerr=[[p - plo], [phi - p]], fmt="none", ecolor=S.DARK[k], capsize=2, elinewidth=0.9, zorder=5)
            ax.plot([xp + 0.45], [p], marker=PERC_MARKER, ms=6, mfc="white", mec=S.DARK[k], mew=1.1, ls="none", zorder=6)
            nlow = int((c["peaks"] < CLIP).sum())
            if nlow:
                ax.text(xp, CLIP - 0.05, "%d" % nlow, ha="center", va="top", fontsize=S.FONT_ROW, color="gray", fontweight="bold")   # legs below the -1 m clip (legend)
            pos.append(xp)
            labels.append("%s\nn = %d" % (S.SHORT[k], len(c["peaks"])))
            if sig == 0.25:
                ax.text(xp, YLIM[0] + 0.02, words[k].replace(", ", ",\n").replace(" (", "\n("), ha="center", va="bottom", fontsize=S.FONT_ROW - (1.5 if S.SLIDE else 0), linespacing=1.1,
                        color=S.GREEN if words[k] == "H1 met" else "k", fontweight="bold" if words[k] == "H1 met" else "normal")
        mid = j * (len(ROWS) + GAP) + (len(ROWS) - 1) / 2
        ax.text(mid, YLIM[1] - 0.12, "σ 0.25: the H1 test (truth graded)" if sig == 0.25 else "σ 0.35: not graded for H1", ha="center", va="top")
    ax.axvline(len(ROWS) - 1 + (1 + GAP) / 2, color="0.35", lw=0.8)
    ax.axhline(S.GUARD_TRIP, color=S.ORANGE, ls=(0, (2, 2)), lw=1.6, zorder=1)
    ax.axhline(S.CONTACT, color=S.RED, lw=1.6, zorder=1)
    ax.axhspan(S.GUARD_TRIP, S.CONTACT, facecolor=S.ORANGE, alpha=0.07, lw=0, zorder=0)   # the 1.42 m built-in margin
    ax.text(pos[0] - 0.58, 3.12, "%.2f m built-in margin between the bar and the contact plane\n(room for the guard to arrest a climb and break away)" % (S.CONTACT - S.GUARD_TRIP), color=S.ORANGE, ha="left", va="center", fontsize=S.FONT_ROW, fontweight="bold")
    ax.text(pos[0] - 0.58, S.GUARD_TRIP + 0.03, "2.00 m: H1 bar", color=S.ORANGE, ha="left", va="bottom", fontsize=S.FONT_ROW, fontweight="bold")
    ax.text(pos[0] - 0.58, S.CONTACT + 0.03, "3.42 m: DHC-2 airframe contact plane", color=S.RED, ha="left", va="bottom", fontsize=S.FONT_ROW, fontweight="bold")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=S.FONT_ROW + 0.5)
    ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.75)
    ax.set_ylim(*YLIM)
    ax.grid(axis="y", alpha=0.3)
    if S.SLIDE:
        ax.set_ylabel("peak upward deviation per leg\n(m above the boom-tip, corridor within 10 m)", fontsize=S.FONT - 2)
    else:
        ax.set_ylabel("per-leg peak upward deviation, meters above the boom-tip\n(terminal approach corridor, within 10 m)")
    h = [Patch(facecolor=S.COLOUR["S1O"], edgecolor=S.DARK["S1O"], label="box = truth peaks per leg: median, IQR,\nwhiskers p5 to p95 (top = the p95), dots beyond"),
         ax.errorbar([np.nan], [np.nan], yerr=[0.1], fmt="_", color="k", ms=8, mew=2.0, capsize=3, elinewidth=0.9, label="truth p95 (graded) with its two-sided\n95% seed-cluster bootstrap interval"),
         Line2D([], [], marker=PERC_MARKER, mfc="white", mec="0.3", color="0.3", ls="none", ms=6, label="perception p95 (sideways diamond) with its\ninterval; not graded (H1 is tested on truth)"),
         Line2D([], [], color="gray", ls="none", marker="$3$", ms=6, label="gray number = legs below the -1 m clip;\nunder each arm: n = legs that entered the corridor"),
         Line2D([], [], color=S.ORANGE, ls=(0, (2, 2)), lw=1.6, label="2.00 m guard trip / H1 bar"),
         Line2D([], [], color=S.RED, lw=1.6, label="3.42 m DHC-2 airframe contact plane")]
    leg = axl.legend(handles=h, loc="center", ncol=2 if S.SLIDE else 3, framealpha=0.95, fontsize=S.FONT_ROW + 0.5, handler_map={type(h[1]): HandlerErrorbar(xerr_size=0.0, yerr_size=0.45)})
    fig.canvas.draw()
    if leg.get_window_extent().width > fig.bbox.width:
        raise SystemExit("REFUSE: legend wider than the figure")
    S.crop_note(fig, axn, 'Per-Leg Peak Upward Deviation, Truth and Perception. Note. Per-leg peak upward deviation above the boom-tip inside the terminal approach corridor (within 10 m), nine wind stages pooled, 540 legs flown per arm and σ, seed block 10775-10804; n = legs that entered the corridor (the rest never came within 10 m and have no corridor peak). Boxes: truth, median, and IQR; whiskers p5 to p95, so the top whisker is the H1 statistic; dots = legs beyond the whiskers; the axis is clipped at -1 m, and legs below are counted in gray. The thick black line is the truth p95 with its two-sided 95% seed-cluster bootstrap interval; the sideways diamond is the perception p95 (what the controller read, same legs) with its interval, not graded. Left group: σ = 0.25, the H1 test; H1 is met when the truth interval lies below 2.00 m (verdict under each box). Right group: σ = 0.35, not graded for H1. Dotted orange = 2.00 m guard trip and H1 bar; solid red = 3.42 m airframe contact plane; the shaded band between them is the 1.42 m built-in margin, the room the guard has to arrest a climb and break away. Dark shades are the arms with the 0.25 m offset (S1O, A2O, PIDO); light shades the same controllers without it.', S.TEXT_W_LANDSCAPE)
    # package 10 (2026-10-02): no in-figure title; the caption carries it
    S.save_exact(fig, STEM, S.TEXT_W_LANDSCAPE)


def table(cells, words):
    rows = []
    for sig in SIGS:
        pid = cells[("PID", sig)]["truth"][0]
        for k in ROWS:
            c = cells[(k, sig)]
            st = box_stats(c["peaks"])
            rows.append({"sigma": "%.2f%s" % (sig, " (H1 test)" if sig == 0.25 else " (not graded for H1)"), "arm": S.SHORT[k],
                         "H1 verdict (bound rule, truth)": words[k] if sig == 0.25 else "", "legs entering the corridor": len(c["peaks"]),
                         "legs below -1 m (clipped)": int((c["peaks"] < CLIP).sum()),
                         "p5 (whisker) / Q1 / median / Q3 / p95 (whisker), truth (m)": "%.3f / %.3f / %.3f / %.3f / %.3f" % (st["whislo"], st["q1"], st["med"], st["q3"], st["whishi"]),
                         "p95, truth (m) [95 %]": "%.3f [%.3f, %.3f]" % c["truth"], "p95, Perception (m) [95 %], not graded": "%.3f [%.3f, %.3f]" % c["perception"],
                         "Perception minus truth (m)": "%+.3f" % float(c["perc_minus_truth"]), "truth p95 minus PID-baseline truth p95 (m)": "%+.3f" % (c["truth"][0] - pid)})
    t = pd.DataFrame(rows)
    stem = os.path.join(S.FIGDIR, STEM)
    t.to_csv(stem + ".csv", index=False)
    notes = ["H1 statistic: corridor p95 of the per-leg peak upward deviation, nine wind stages pooled, 540 legs per arm, seed block 10775-10804; legs that never entered the corridor are nulls.",
             "H1 is graded on sigma 0.25, truth (the table “The Five FSM States That Decide an Aerial-Docking Leg”): met when the upper end of the two-sided 95%% seed-cluster bootstrap interval of the p95 lies below %.2f m. Verdict words from PVALUE_BOOTSTRAP_20260922.csv. Sigma 0.35 is not graded for H1." % S.GUARD_TRIP,
             "Box statistics are over the legs that entered the corridor; whiskers p5 to p95. n = legs, of 540, that entered the corridor; the rest never came within 10 m of the boom-tip. Intervals: two-sided 95 % seed-cluster bootstrap (p95_boot.py).",
             "Perception = what the controller reads (ctrl_*), same legs, same truth-selected ticks; not graded."]
    open(stem + ".md", "w").write("# H1 per-leg corridor peaks by arm, truth and Perception - APPENDIX; POST-PLAN PRESENTATION; decides nothing\n\nSuggested figure title: " + TITLE + "\n\nFigure description (for the document, not drawn in the figure): " + DESC_NOTE + "\n\n"
                                  + S.md(t) + "\n\n" + "\n".join("- " + x for x in notes) + "\n")
    print("wrote", stem + ".csv|.md")


def main():
    cells, words = load()
    figure(cells, words)
    table(cells, words)
    return 0


if __name__ == "__main__":
    sys.exit(main())
