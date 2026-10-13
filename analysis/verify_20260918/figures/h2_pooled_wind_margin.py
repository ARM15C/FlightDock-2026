#!/usr/bin/env python3
# =============================================================================

#   decides nothing. Verdict words follow the table “The Five FSM States That Decide an Aerial-Docking Leg” (v3): H2 is MET when the two-sided 95% seed-cluster bootstrap
#   interval of the same-seed paired difference in docking-envelope captures (arm minus PID-baseline, sigma 0.25, nine wind
#   stages pooled, 540 pairs) lies above 0 pp.
# v6 LAYOUT: ONE panel, the nine-stage pool (THE H2 TEST), built at the PORTRAIT PRINT WIDTH
#   (figure_style.TEXT_W_PORTRAIT, 6.0 in) with 8 pt text so Word never scales it. The strong-wind pool (180 pairs,
#   comparator (not graded)) moved to the figure strong_wind_pools_appendix.py, which imports draw_panel() from here.
#   Six rows S1O, S1, A2O, A2, PIDO, PID (PID at 0 pp, always drawn). Marker = paired difference (hollow = with the 0.25 m
#   offset, filled = without); whisker = two-sided 95% seed-cluster bootstrap interval (whole seeds, 10,000 resamples, fixed
#   seed: hypothesis_alternatives_v4.boot_diff). Dotted green line at 0 pp, labelled. Two-line row labels: value and interval,
#   then "H2 met" / "H2 not met" / "comparator (not graded)" / "baseline". x range fitted to the measured text; ticks every 5 pp.
#   Legend inside the axes if it overlaps nothing, else below. No description text in the figure (it is in the .md).
# SOURCES (nothing typed): scoring board via pvalue_supplement.board(); verdict tool paired(); hypothesis_alternatives_v4.boot_diff.
# VALIDATION, ENFORCED (refuses): differences, pairs, discordant counts == comparative/ARM_PAIRS.csv; bootstrap intervals ==
#   whatif_20260920/H2_ALTERNATIVES_v4 (0.1 dp) and, nine-stage pool, == pvalue_bootstrap_20260922/PVALUE_BOOTSTRAP_20260922.csv
#   (2 dp) with its A_interval_clears flag; seed block 10775-10804; 540 and 180 pairs. The PNG width is checked in pixels.
# CLI: (none) SIDE-EFFECTS: writes H2_POOLED_WIND_MARGIN.png/.pdf/.csv/.md under OUT/figures.
# =============================================================================
# 2026-10-01 ADVISOR RULING: POOLED WINDS ONLY. The strong-wind pool, strong_wind_pools_appendix.py and their
#   outputs are ARCHIVED in archive_strong_wind_20261001/; older comments below that mention the figure are historical.
#   WORDING: "comparator (not graded)" replaces "reported, not graded" for the PID-baseline and PIDO; the PID H2 row reads
#   "baseline comparator (not graded)"; sigma 0.35 and other ungraded readings say "not graded for H1".
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import os
import sys

import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figure_style as S  # noqa: E402
sys.path.insert(0, (_REPO + "/analysis/whatif_20260920"))
import hypothesis_alternatives_v4 as HA  # noqa: E402

plt = S.plt
V = S.V
STEM = "H2_POOLED_WIND_MARGIN"
GREEN = "#2ca02c"
ROWS = list(S.ROW_ORDER)
SIG = 0.25
BAR = 0.0
WHATIF = (_REPO + "/analysis/whatif_20260920/H2_ALTERNATIVES_v4_20260920.csv")
PVAL = (_REPO + "/analysis/pvalue_bootstrap_20260922/PVALUE_BOOTSTRAP_20260922.csv")
# pool labels: ARM_PAIRS.csv writes "strong-wind (6-8 m/s)", H2_ALTERNATIVES_v4 writes "strong wind (6-8 m/s)"; both carried
POOLS = [("nine", "nine wind stages", "nine wind stages", "wind_rung", "Nine wind stages pooled (0 to 8 m/s, σ=0.25)\n540 same-seed pairs per arm: the H2 test", 540)]
LABEL_OF = {tag: label for tag, _, _, _, label, _ in POOLS}
TITLE = "H2 Margin: Same-Seed Difference in Docking-Envelope Captures Against the PID-Baseline, Light Turbulence (σ = 0.25)"
DESC = ("Paired difference in docking-envelope captures, each arm minus the PID-baseline on the same seed, wind stage and trajectory, "
        "under light turbulence (σ = 0.25), nine wind stages pooled, 540 pairs per arm, seed block 10775-10804. Marker = the difference in "
        "percentage points; whisker = its two-sided 95% seed-cluster bootstrap interval (30 seeds resampled whole, 10,000 draws). The dotted "
        "green line at 0 pp is the H2 bar and the PID-baseline. H2 is met when the whole interval lies above 0 pp. PIDO is a comparator (not graded). "
        "Right is better. Hollow markers and dark shades are the arms with the 0.25 m offset (S1O, A2O, PIDO); filled markers and light shades "
        "are the same controllers without it (S1, A2, PID).")


def load():
    bd = HA.P.board()
    for c, g in bd.groupby("corpus"):
        if sorted(g.seed.unique()) != S.SEED_BLOCK:
            raise SystemExit("REFUSE: corpus %s is not seed block 10775-10804" % c)
    pairs = pd.read_csv(os.path.join(S.RUN, "comparative", "ARM_PAIRS.csv"))
    wi = pd.read_csv(WHATIF)
    pv = pd.read_csv(PVAL)
    pv = pv[pv.hypothesis == "H2"].set_index("arm")
    cells, words = {}, {}
    for tag, pool, wpool, col, _, n in POOLS:
        for k in ROWS:
            if k == "PID":
                cells[(k, tag)] = dict(n=n, ka=None, kb=None, diff=0.0, b=None, c=None, two=(None, None), p_one=None)
                continue
            r = V.paired(bd, V.corpus(k, SIG), V.corpus("PID", SIG), lambda d, c=col: d[c])
            a, b = (k, "PID") if S.ARMS.index(k) < S.ARMS.index("PID") else ("PID", k)
            p = pairs[(pairs.sigma == SIG) & (pairs.arm_a == V.NAME[a]) & (pairs.arm_b == V.NAME[b]) & (pairs.pool == pool)]
            sign = 1 if a == k else -1
            if len(p) != 1 or r["n"] != n or int(p.n_pairs.iloc[0]) != n or round(sign * float(p.diff_pp.iloc[0]), 2) != round(r["diff"], 2):
                raise SystemExit("REFUSE: %s %s does not reproduce ARM_PAIRS.csv" % (k, pool))
            disc = "%d / %d" % ((r["b"], r["c"]) if sign == 1 else (r["c"], r["b"]))
            if disc != p.discordant.iloc[0]:
                raise SystemExit("REFUSE: %s %s discordant pairs differ from ARM_PAIRS.csv" % (k, pool))
            lo, hi = HA.boot_diff(bd, r, V.corpus(k, SIG))
            w = wi[(wi.sigma == SIG) & (wi.pool == wpool) & (wi.arm == V.NAME[k])]
            if len(w) != 1 or (round(lo, 1), round(hi, 1)) != (float(w.boot_lo.iloc[0]), float(w.boot_hi.iloc[0])) or round(r["diff"], 2) != float(w.diff_pp.iloc[0]):
                raise SystemExit("REFUSE: %s %s bootstrap interval does not reproduce H2_ALTERNATIVES_v4" % (k, pool))
            pone = None
            if tag == "nine":
                q = pv.loc[k]
                if (round(lo, 2), round(hi, 2), round(r["diff"], 2)) != (float(q.ci95_lo), float(q.ci95_hi), float(q.value)):
                    raise SystemExit("REFUSE: %s nine-stage interval does not reproduce PVALUE_BOOTSTRAP_20260922.csv" % k)
                met = lo > BAR
                if bool(q.A_interval_clears) != met:
                    raise SystemExit("REFUSE: %s bound-rule verdict differs from PVALUE_BOOTSTRAP A_interval_clears" % k)
                pone = float(q.boot_p_one_sided)
                if k in V.POL:
                    words[k] = "H2 met" if met else "H2 not met"
            cells[(k, tag)] = dict(r, two=(lo, hi), p_one=pone)
    print("VALIDATION PASS: 5 comparisons (5 arms, nine-stage pool, sigma 0.25) reproduce ARM_PAIRS.csv and H2_ALTERNATIVES_v4; nine-stage intervals and verdict flags reproduce PVALUE_BOOTSTRAP_20260922.csv")
    return cells, words


X_TEXT_GAP = 1.5      # pp between the widest whisker end and the start of the row text


def row_text(ax, x, y, txt):
    """Row label; a second line reading "H2 met" is drawn in green. Returns the artist whose extent is widest."""
    lines = txt.split("\n")
    if len(lines) == 2 and lines[1] == "H2 met":
        a = ax.text(x, y, lines[0] + "\n", va="center", ha="left", linespacing=1.2, fontsize=S.FONT_ROW)
        ax.text(x, y, "\n" + lines[1], va="center", ha="left", linespacing=1.2, fontsize=S.FONT_ROW, color=S.GREEN, fontweight="bold")
        return a
    return ax.text(x, y, txt, va="center", ha="left", linespacing=1.2, fontsize=S.FONT_ROW)


def draw_panel(fig, ax, cells, words, tag, xlabel=None, one_line_title=True, tick_step=5):
    """One margin panel: rows, whiskers, markers, two-line labels (one line off the nine-stage pool), bar line, fitted x range."""
    sub = {k: cells[(k, tag)] for k in ROWS}
    lo = min(c["two"][0] for c in sub.values() if c["two"][0] is not None)
    hi = max(c["two"][1] for c in sub.values() if c["two"][1] is not None)
    texts, artists = [], []
    for j, k in enumerate(ROWS):
        c, y = sub[k], len(ROWS) - 1 - j
        if k != "PID":
            artists += ax.plot(list(c["two"]), [y, y], color=S.whisker_colour(k), lw=S.LW_WHISKER, solid_capstyle="butt", zorder=3)
            artists += ax.plot(list(c["two"]), [y, y], marker="|", ms=8, mew=S.LW_WHISKER * 0.8, color=S.whisker_colour(k), ls="none", zorder=3)
        artists += ax.plot([c["diff"]], [y], marker=S.MARKER[k], ms=S.MS, mfc=S.marker_face(k), mec=S.DARK[k], mew=1.4, ls="none", zorder=4)
        if k == "PID":
            txt = "0.00 pp\nbaseline comparator (not graded)"
        else:
            txt = "%+.2f pp [%+.1f, %+.1f]" % (c["diff"], c["two"][0], c["two"][1])
            if tag == "nine":
                txt += "\n" + (words[k] if k in words else "comparator (not graded)")
        texts.append(row_text(ax, hi + X_TEXT_GAP, y, txt))
    ax.axvline(BAR, color=GREEN, lw=2.0, ls=(0, (2, 2)), zorder=1)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([S.SHORT[k] for k in reversed(ROWS)])
    ax.set_ylim(-0.6, len(ROWS) - 0.15)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel(xlabel or "difference in captures against the PID-baseline (pp); right is better")   # package 10: plain label; no in-figure title (the caption carries it)
    artists.append(ax.text(BAR + 0.8, len(ROWS) - 0.25, "0 pp: H2 bar = PID baseline", color=GREEN, ha="left", va="top", fontweight="bold", fontsize=S.FONT_ROW, zorder=6))
    xmin = lo - 4
    ax.set_xlim(xmin, hi + 52)   # provisional
    xmax = S.fit_text_edge(fig, ax, texts, xmin, hi + X_TEXT_GAP)
    ax.set_xticks([t for t in range(-50, 101, tick_step) if xmin <= t <= hi + 3])   # no tick under the text column
    return texts + artists


def legend_handles():
    return [Line2D([], [], marker="s", color=S.DARK["S1O"], mfc=S.marker_face("S1O"), mec=S.DARK["S1O"], ls="none", ms=8, label="marker = paired difference;\nhollow = with the 0.25 m offset"),
            Line2D([], [], marker="s", color=S.COLOUR["S1"], mfc=S.marker_face("S1"), mec=S.DARK["S1"], ls="none", ms=8, label="filled = without the offset"),
            Line2D([], [], color="0.3", lw=S.LW_WHISKER, marker="|", ms=10, mew=2.2, label="whisker = two-sided 95%\nseed-cluster bootstrap interval"),
            Line2D([], [], color=GREEN, lw=2.0, ls=(0, (2, 2)), label="0 pp = H2 bar = PID baseline")]


FIGURE_NOTE = ("Marker = the paired difference; hollow = with the 0.25 m offset (S1O, A2O, PIDO), filled = without (S1, A2, PID). "
               "Whisker = two-sided 95% seed-cluster bootstrap interval. Dotted green line = 0 pp, the H2 bar and the PID-baseline.")


NOTE_IN = 1.25   # inches reserved under the chart for the crop note


def figure(cells, words):
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(S.TEXT_W_PORTRAIT, 2.6 + NOTE_IN), gridspec_kw={"height_ratios": [2.6, NOTE_IN], "hspace": 0.02})
    fig.set_layout_engine("constrained")
    draw_panel(fig, ax, cells, words, "nine")          # no legend in the figure
    S.crop_note(fig, axn, "Title: " + TITLE + " || Description: " + DESC + " || Legend: " + FIGURE_NOTE, S.TEXT_W_PORTRAIT)
    S.save_exact(fig, STEM, S.TEXT_W_PORTRAIT)


def table(cells, words):
    rows = []
    for tag, _, _, _, label, _ in POOLS:
        for k in ROWS:
            c = cells[(k, tag)]
            if k == "PID":
                rows.append({"arm": S.SHORT[k], "H2 verdict (bound rule, nine stages)": "baseline comparator (not graded)" if tag == "nine" else "",
                             "same-seed pairs": c["n"], "docking-envelope captures, arm": "", "docking-envelope captures, PID-baseline (comparator)": "",
                             "paired difference (pp)": "0.00", "two-sided 95% seed-cluster bootstrap interval (pp)": "", "one-sided bootstrap p (H0: difference <= 0)": "",
                             "pairs only the arm captured / only the PID-baseline captured": ""})
                continue
            rows.append({"arm": S.SHORT[k],
                         "H2 verdict (bound rule, nine stages)": (words.get(k, "comparator (not graded)") if tag == "nine" else ""),
                         "same-seed pairs": c["n"],
                         "docking-envelope captures, arm": "%d (%.1f%%)" % (c["ka"], 100.0 * c["ka"] / c["n"]),
                         "docking-envelope captures, PID-baseline (comparator)": "%d (%.1f%%)" % (c["kb"], 100.0 * c["kb"] / c["n"]),
                         "paired difference (pp)": "%+.2f" % c["diff"],
                         "two-sided 95% seed-cluster bootstrap interval (pp)": "[%+.2f, %+.2f]" % c["two"],
                         "one-sided bootstrap p (H0: difference <= 0)": ("%.4f" % c["p_one"]) if c["p_one"] is not None else "",
                         "pairs only the arm captured / only the PID-baseline captured": "%d / %d" % (c["b"], c["c"])})
    t = pd.DataFrame(rows)
    stem = os.path.join(S.FIGDIR, STEM)
    t.to_csv(stem + ".csv", index=False)
    notes = ["Pairs are (stage, seed, trajectory) at sigma 0.25; seed block 10775-10804; 30 seeds. Success = the four-condition flag of record (the document).",
             "H2 rule (the table “The Five FSM States That Decide an Aerial-Docking Leg”): met when the two-sided 95% seed-cluster bootstrap interval of the nine-stage paired difference lies above 0 pp (one-sided test at alpha 0.025). Whole seeds are resampled, 10,000 draws, fixed random seed; the 2.5th and 97.5th percentiles bound the interval.",
             "PIDO is a comparator: reported with the same statistic, not graded. The PID-baseline is the baseline comparator in every row and is drawn at 0 pp.",
             "Palette: " + S.PALETTE_RULE + "."]
    open(stem + ".md", "w").write("# H2 pooled-wind margin - POST-PLAN PRESENTATION; decides nothing\n\n"
                                  "Suggested figure title: " + TITLE + "\n\nFigure description (for the document, not drawn in the figure): " + DESC + "\n\nFigure note (the legend, for your 9 pt note under the figure): " + FIGURE_NOTE + "\n\n"
                                  + S.md(t) + "\n\n" + "\n".join("- " + x for x in notes) + "\n")
    print("wrote", stem + ".csv|.md")


def main():
    cells, words = load()
    figure(cells, words)
    table(cells, words)
    return 0


if __name__ == "__main__":
    sys.exit(main())
