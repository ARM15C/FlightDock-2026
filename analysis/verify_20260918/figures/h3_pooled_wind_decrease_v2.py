#!/usr/bin/env python3
# =============================================================================

#   version (Newcombe whisker, point rule; archived in archive_h3_newcombe_point_rule_20261001/) to the the document v3 rules.
#   POST-PLAN PRESENTATION; decides nothing. Verdict words follow the table “The Five FSM States That Decide an Aerial-Docking Leg” (v3): H3 is MET when the upper end of
#   the two-sided 95% seed-cluster bootstrap interval of the same-seed decrease in docking-envelope captures (sigma 0.25 rate
#   minus sigma 0.35 rate, same arm, nine wind stages pooled, 540 pairs) lies below 35 pp.
# v2 LAYOUT: ONE panel, the nine-stage pool (THE H3 TEST), at the PORTRAIT PRINT WIDTH (6.0 in),
#   8 pt text. The strong-wind pool (180 pairs, comparator (not graded)) moved to strong_wind_pools_appendix.py, which imports
#   draw_panel() from here. Six rows S1O, S1, A2O, A2, PIDO, PID. Marker = paired decrease; whisker = two-sided 95% seed-cluster
#   bootstrap interval (hypothesis_alternatives_v4.boot_diff). Dotted green line at 35 pp = the H3 bar, labelled. The x-axis is
#   REVERSED so that right = smaller decrease = better; the axis label says so. No zero
#   line. Two-line row labels; PID and PIDO read "comparator (not graded)" (only FlightDock controllers are graded). x range fitted
#   to the measured text; ticks every 5 pp, none under the text. Legend inside the axes if it overlaps nothing, else below.
# PALETTE: figure_style (2026-10-01 v4): dark shade + hollow marker = with the 0.25 m offset; light shade + filled = without.
# SOURCES (nothing typed): scoring board via pvalue_supplement.board(); verdict tool paired(); hypothesis_alternatives_v4.boot_diff.
# VALIDATION, ENFORCED (refuses): decreases and intervals == whatif_20260920/H3_ALTERNATIVES_v4 (diff 2 dp, interval 0.1 dp,
#   counts, judged flag); nine-stage pool == pvalue_bootstrap_20260922/PVALUE_BOOTSTRAP_20260922.csv H3 rows (2 dp) and its
#   A_interval_clears flag; seed block 10775-10804; 540 and 180 pairs.
# CLI: (none) SIDE-EFFECTS: writes H3_POOLED_WIND_DECREASE.png/.pdf/.csv/.md under OUT/figures.
# =============================================================================
# 2026-10-01 ADVISOR RULING: POOLED WINDS ONLY. The strong-wind pool, strong_wind_pools_appendix.py and their
#   outputs are ARCHIVED in archive_strong_wind_20261001/; older comments below that mention the figure are historical.
#   WORDING: "comparator (not graded)" replaces "reported, not graded" for the PID-baseline and PIDO; the PID H2 row reads
#   "baseline comparator (not graded)"; sigma 0.35 and other ungraded readings say "not graded for H1".
# 2026-10-02 v2:
#   same frame as H2_POOLED_WIND_MARGIN_v2: rates above each marker ('67.2% to 40.2%'), the decrease below it (positive number, the
#   the table “H3: Capture Decrease, Light to Moderate Turbulence” convention), interval ends under the whisker caps, verdict word only at the right (red when not met), fonts +0.5 pt,
#   arm labels bold in the family colour, two-line x label, red arrow at the top left = direction of decreasing capture success,
#   2026-10-02: EVERY TRACE OF THE 35% FLOOR REMOVED from this chart (ticks, label, note, csv column); the floor at
#   sigma 0.35 is drawn on capture_margin_against_pid_sigma035.py instead. Comparator verdicts on two lines. (Superseded lines follow:
#   sigma 0.25 rate minus 35, computed from the captures of record and checked against results_tables/RESULTS_NUMBERS.json FLOOR rates).
#   The floor comparison text lines of v1 are dropped. Every added label is measured; the script refuses if two overlap.
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import os
import sys

import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figure_style as S  # noqa: E402
sys.path.insert(0, (_REPO + "/analysis/whatif_20260920"))
import hypothesis_alternatives_v4 as HA  # noqa: E402
import p95_boot as PB  # noqa: E402
import numpy as np  # noqa: E402
FLOOR = 35.0   # minimum capture rate of reference: read against the pooled sigma 0.35 rate; reported, not a verdict

plt = S.plt
V = S.V
S.FONT, S.FONT_ROW, S.FONT_TITLE_MAX = S.FONT + 0.5, S.FONT_ROW + 0.5, S.FONT_TITLE_MAX + 0.5   # v2: +0.5 pt everywhere (crop_note reads S.FONT_ROW)
plt.rcParams.update({"font.size": S.FONT, "axes.labelsize": S.FONT, "axes.titlesize": S.FONT, "xtick.labelsize": S.FONT, "ytick.labelsize": S.FONT,
                     "legend.fontsize": S.FONT_ROW + 0.5 if S.SLIDE else S.FONT})
STEM = "H3_POOLED_WIND_DECREASE_v2"
GREEN = "#2ca02c"
ROWS = list(S.ROW_ORDER)
BAR = float(V.H3_BAR)
WHATIF = (_REPO + "/analysis/whatif_20260920/H3_ALTERNATIVES_v4_20260920.csv")
PVAL = (_REPO + "/analysis/pvalue_bootstrap_20260922/PVALUE_BOOTSTRAP_20260922.csv")
PV_ARM = {"PID": "PID-baseline"}   # PVALUE_BOOTSTRAP writes the PID-baseline row under its full name
POOLS = [("nine", "nine wind stages", "wind_rung", "Nine wind stages pooled (0 to 8 m/s), σ 0.25 paired with σ 0.35 on the same seeds\n540 same-seed pairs per arm: the H3 test", 540)]
LABEL_OF = {tag: label for tag, _, _, label, _ in POOLS}
TITLE = "H3 Margin: Same-Seed Decrease in Docking-Envelope Captures From Light (σ = 0.25) to Moderate (σ = 0.35) Turbulence"
NOTE = ("Marker = each arm's same-seed decrease in docking-envelope captures from light (σ = 0.25) to moderate (σ = 0.35) turbulence, nine wind "
        "stages pooled, printed as a positive number; whisker = its interval, ends labeled; the capture rates at both levels are printed above the "
        "marker. The axis is reversed, so a larger decrease lies to the left and right is better. Dotted green line = 35 pp, the H3 bar. Hollow markers and dark shades = with the 0.25 m offset (S1O, A2O, PIDO); filled markers and "
        "light shades = without it (S1, A2, PID).")
def load():
    bd = HA.P.board()
    for c, g in bd.groupby("corpus"):
        if sorted(g.seed.unique()) != S.SEED_BLOCK:
            raise SystemExit("REFUSE: corpus %s is not seed block 10775-10804" % c)
    wi = pd.read_csv(WHATIF)
    pv = pd.read_csv(PVAL)
    pv = pv[pv.hypothesis == "H3"].set_index("arm")
    cells, words = {}, {}
    for tag, wpool, col, _, n in POOLS:
        for k in ROWS:
            a = V.corpus(k, 0.25)
            r = V.paired(bd, a, V.corpus(k, 0.35), lambda d, c=col: d[c])
            if r["n"] != n:
                raise SystemExit("REFUSE: %s %s has %d pairs, not %d" % (k, tag, r["n"], n))
            lo, hi = HA.boot_diff(bd, r, a)
            w = wi[(wi.pool == wpool) & (wi.arm == V.NAME[k])]
            if (len(w) != 1 or round(r["diff"], 2) != float(w.decrease_pp.iloc[0]) or (round(lo, 1), round(hi, 1)) != (float(w.boot_lo.iloc[0]), float(w.boot_hi.iloc[0]))
                    or (r["ka"], r["kb"]) != (int(w.captures_s025.iloc[0]), int(w.captures_s035.iloc[0])) or bool(w.judged.iloc[0]) != (k in V.POL)):
                raise SystemExit("REFUSE: %s %s does not reproduce H3_ALTERNATIVES_v4" % (k, tag))
            pone = None
            if tag == "nine":
                q = pv.loc[PV_ARM.get(k, k)]
                if (round(lo, 2), round(hi, 2), round(r["diff"], 2)) != (float(q.ci95_lo), float(q.ci95_hi), float(q.value)) or float(q.bar) != BAR:
                    raise SystemExit("REFUSE: %s nine-stage interval does not reproduce PVALUE_BOOTSTRAP_20260922.csv" % k)
                met = hi < BAR
                if bool(q.A_interval_clears) != met:
                    raise SystemExit("REFUSE: %s bound-rule verdict differs from PVALUE_BOOTSTRAP A_interval_clears" % k)
                pone = float(q.boot_p_one_sided)
                words[k] = ("H3 met" if met else "H3 not met") if k in V.POL else "comparator (not graded)"
            g = bd[(bd.corpus == V.corpus(k, 0.35)) & bd[col]]
            ok, seed = g.passed.astype(str).str.lower().eq("true").to_numpy(), g.seed.to_numpy(); labels = np.sort(np.unique(seed))
            if len(g) != n or int(ok.sum()) != r["kb"] or list(labels) != S.SEED_BLOCK:
                raise SystemExit("REFUSE: %s sigma 0.35 pooled captures do not reproduce the paired count" % k)
            ks = np.array([ok[seed == x].sum() for x in labels], float); ns = np.array([(seed == x).sum() for x in labels], float); picks = PB._picks(len(labels))
            vals = 100.0 * ks[picks].sum(axis=1) / ns[picks].sum(axis=1)
            flo, fhi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)); frate = 100.0 * ok.mean()
            cells[(k, tag)] = dict(r, two=(lo, hi), p_one=pone, floor=(frate, flo, fhi, "above" if flo > FLOOR else ("below" if fhi < FLOOR else "straddles")))
    print("VALIDATION PASS: 6 decreases (6 arms, nine-stage pool) reproduce H3_ALTERNATIVES_v4; nine-stage intervals and verdict flags reproduce PVALUE_BOOTSTRAP_20260922.csv; words %s" % words)
    return cells, words


X_TEXT_GAP = 1.5      # pp between the smallest whisker end and the start of the verdict column (toward the right on the reversed axis)
DY = 0.27             # rows: label offset above / below the marker line (data units, one row = 1)
YPAD = 1.05           # room above the top row for the bar label, the floor label and the direction arrow
BOX = dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none")   # white backing so a number stays legible over a rule line or the band


def row_text(ax, x, y, txt):
    """Verdict word at the right of the row: "H3 met" green bold, "H3 not met" red, comparators black."""
    met, notmet = txt == "H3 met", txt == "H3 not met"
    return ax.text(x, y, txt, va="center", ha="left", fontsize=S.FONT_ROW, linespacing=1.15, color=S.GREEN if met else (S.RED if notmet else "k"), fontweight="bold" if met else "normal")


def no_overlap(fig, labels, what):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bbs = [(t, t.get_window_extent(r)) for t in labels]
    for a in range(len(bbs)):
        for b in range(a + 1, len(bbs)):
            if bbs[a][1].overlaps(bbs[b][1]):
                raise SystemExit("REFUSE: %s labels overlap: %r and %r" % (what, bbs[a][0].get_text(), bbs[b][0].get_text()))


def draw_panel(fig, ax, cells, words, tag, xlabel=None, one_line_title=True, tick_step=5):
    """v2 decrease panel on a REVERSED axis, in the frame of the H2 v2 chart."""
    sub = {k: cells[(k, tag)] for k in ROWS}
    lo = min(c["two"][0] for c in sub.values())
    hi = max(c["two"][1] for c in sub.values())
    x_left, x_text = hi + 3, lo - X_TEXT_GAP
    texts, artists, labels = [], [], []
    for j, k in enumerate(ROWS):
        c, y = sub[k], len(ROWS) - 1 - j
        artists += ax.plot(list(c["two"]), [y, y], color=S.whisker_colour(k), lw=S.LW_WHISKER, solid_capstyle="butt", zorder=3)
        artists += ax.plot(list(c["two"]), [y, y], marker="|", ms=8, mew=S.LW_WHISKER * 0.8, color=S.whisker_colour(k), ls="none", zorder=3)
        artists += ax.plot([c["diff"]], [y], marker=S.MARKER[k], ms=S.MS, mfc=S.marker_face(k), mec=S.DARK[k], mew=1.4, ls="none", zorder=4)
        for xe in c["two"]:
            labels.append(ax.text(xe, y - DY, "%.2f" % xe, ha="center", va="top", fontsize=S.FONT_ROW, bbox=BOX, zorder=6))
        labels.append(ax.text(c["diff"], y + DY, "%.1f%% to %.1f%%" % (100.0 * c["ka"] / c["n"], 100.0 * c["kb"] / c["n"]), ha="center", va="bottom", fontsize=S.FONT_ROW, fontweight="bold", bbox=BOX, zorder=6))
        labels.append(ax.text(c["diff"], y - DY, "%.2f pp" % c["diff"], ha="center", va="top", fontsize=S.FONT_ROW, bbox=BOX, zorder=6))
        texts.append(row_text(ax, x_text, y, words[k].replace(" (", "\n(")))   # comparators on two lines
    ax.axvline(BAR, color=GREEN, lw=2.0, ls=(0, (2, 2)), zorder=1)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([S.SHORT[k] for k in reversed(ROWS)])
    for t, k in zip(ax.get_yticklabels(), reversed(ROWS)):   # v2: arm name bold in the family's dark colour
        t.set_color(S.DARK[k]); t.set_fontweight("bold")
    ax.set_ylim(-0.75, len(ROWS) - 1 + YPAD)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_xlabel(xlabel or "Docking-envelope capture rate difference (pp)")   # v2; second line drawn below at row size
    ax.annotate("controller's performance compared against itself as turbulence increases from light (σ = 0.25) to moderate (σ = 0.35)",
                xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -(S.FONT * 3.4)), textcoords="offset points", ha="center", va="top", fontsize=S.FONT_ROW)
    ytop = len(ROWS) - 1 + YPAD - 0.05
    labels.append(ax.text(BAR - 0.8, ytop, "35 pp: H3 bar", color=GREEN, ha="left", va="top", fontweight="bold", fontsize=S.FONT_ROW, zorder=6))
    xa_head, xa_tail = x_left - 0.6, x_left - 9.0   # the arrow points LEFT (larger decreases) on the reversed axis
    ax.annotate("", xy=(xa_head, ytop - 0.1), xytext=(xa_tail, ytop - 0.1), arrowprops=dict(arrowstyle="->", color=S.RED, lw=1.6, shrinkA=0, shrinkB=0), zorder=6)
    labels.append(ax.text((xa_head + xa_tail) / 2, ytop - 0.22, "decreasing docking-envelope\ncapture success rate", color=S.RED, ha="center", va="top", fontsize=S.FONT_ROW, linespacing=1.2, zorder=6))
    ax.set_xlim(x_left, x_text - 40)   # provisional, REVERSED
    S.fit_text_edge(fig, ax, texts, x_left, x_text)
    ax.set_xticks([t for t in range(0, 101, tick_step) if x_text + 1 <= t <= x_left])   # no tick under the text column
    no_overlap(fig, labels, "in-plot")
    return texts + artists + labels


def legend_handles():
    return [Line2D([], [], marker="s", color=S.DARK["S1O"], mfc=S.marker_face("S1O"), mec=S.DARK["S1O"], ls="none", ms=8, label="marker = paired decrease;\nhollow = with the 0.25 m offset"),
            Line2D([], [], marker="s", color=S.COLOUR["S1"], mfc=S.marker_face("S1"), mec=S.DARK["S1"], ls="none", ms=8, label="filled = without the offset"),
            Line2D([], [], color="0.3", lw=S.LW_WHISKER, marker="|", ms=10, mew=2.2, label="whisker = two-sided 95%\nseed-cluster bootstrap interval"),
            Line2D([], [], color=GREEN, lw=2.0, ls=(0, (2, 2)), label="35 pp = H3 bar")]


NOTE_IN = 0.75   # v2: shorter note


def figure(cells, words):
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(S.TEXT_W_PORTRAIT, 3.7 + NOTE_IN), gridspec_kw={"height_ratios": [3.7, NOTE_IN], "hspace": 0.02})   # v2: taller rows and label strip
    fig.set_layout_engine("constrained")
    draw_panel(fig, ax, cells, words, "nine")          # no legend in the figure
    S.crop_note(fig, axn, 'H3: Capture Decrease, Light to Moderate Turbulence. Note. Capture = truth and perception both inside M3b for 3.0 s (the “Decision Rules” section). Marker: each arm’s same-seed decrease in docking-envelope captures from light (σ = 0.25) to moderate (σ = 0.35) turbulence. Nine wind stages pooled for both turbulence levels and compared, printed as positive number. Hollow markers and dark shades: includes 0.25 m PTA offset (S1O, A2O, PIDO). Filled markers and light shades: does not include PTA (S1, A2, PID). Whiskers: its CI, ends labeled. Capture rates at both turbulence levels printed above the marker. Dotted green line: H3 bar of 35 pp. X-axis reversed, so larger decrease in captures are to the left, right of green line means capture rate success is degrading less in an atmospheric environment with 40% increase in gust intensities.', S.TEXT_W_PORTRAIT)
    S.save_exact(fig, STEM, S.TEXT_W_PORTRAIT)


def table(cells, words):
    rows = []
    for tag, _, _, label, _ in POOLS:
        for k in ROWS:
            c = cells[(k, tag)]
            rows.append({"arm": S.SHORT[k], "H3 verdict (bound rule, nine stages)": words[k] if tag == "nine" else "",
                         "same-seed pairs": c["n"],
                         "docking-envelope captures, sigma 0.25": "%d (%.1f%%)" % (c["ka"], 100.0 * c["ka"] / c["n"]),
                         "docking-envelope captures, sigma 0.35": "%d (%.1f%%)" % (c["kb"], 100.0 * c["kb"] / c["n"]),
                         "decrease (pp)": "%.2f" % c["diff"],
                         "two-sided 95% seed-cluster bootstrap interval (pp)": "[%.2f, %.2f]" % c["two"],
                         "one-sided bootstrap p (H0: decrease >= 35 pp)": ("%.4f" % c["p_one"]) if c["p_one"] is not None else "",
                         "pairs captured at sigma 0.25 only / at sigma 0.35 only": "%d / %d" % (c["b"], c["c"])})
    t = pd.DataFrame(rows)
    stem = os.path.join(S.FIGDIR, STEM)
    t.to_csv(stem + ".csv", index=False)
    notes = ["Pairs are (stage, seed, trajectory), the same arm at sigma 0.25 and at sigma 0.35; seed block 10775-10804; 30 seeds. Success = the four-condition flag of record (the document).",
             "H3 rule (the table “The Five FSM States That Decide an Aerial-Docking Leg”): met when the upper end of the two-sided 95% seed-cluster bootstrap interval of the nine-stage decrease lies below 35 pp (one-sided test at alpha 0.025). Whole seeds are resampled, 10,000 draws, fixed random seed; the 2.5th and 97.5th percentiles bound the interval.",
             "Only the FlightDock controllers are graded. The PID-baseline and PIDO are comparators: reported with the same statistic, not graded.",
             "The x-axis of the figure is reversed so that right is better, the same reading direction as the H2 figure. The 35% reference floor is not drawn on this chart; see capture_margin_against_pid_sigma035.py and the reference-floor table.",
             "Palette: " + S.PALETTE_RULE + "."]
    open(stem + ".md", "w").write("# H3 pooled-wind decrease, v2 (2026-10-02) - POST-PLAN PRESENTATION; decides nothing\n\n"
                                  "Suggested figure title: " + TITLE + "\n\nFigure note (for your note under the figure; the pairs, seed block, bootstrap and bar are in the table “H3: Capture Decrease, Light to Moderate Turbulence” note): " + NOTE + "\n\n"
                                  + S.md(t) + "\n\n" + "\n".join("- " + x for x in notes) + "\n")
    print("wrote", stem + ".csv|.md")


def main():
    cells, words = load()
    figure(cells, words)
    table(cells, words)
    return 0


if __name__ == "__main__":
    sys.exit(main())
