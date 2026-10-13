#!/usr/bin/env python3
# =============================================================================

#   moderate turbulence side by side. POST-PLAN PRESENTATION; decides nothing.
# WHAT IS DRAWN: landscape 9.0 in, two stacked panels (sigma 0.25 top, sigma 0.35 bottom). x = the seven
#   funnel steps of the table; bar height = share of the 540 wind-stage legs that reached the step at least once, bars in the arm colours of record
#   (dark = with the 0.25 m offset), black capped whisker = two-sided 95% seed-cluster bootstrap interval. EVERY BAR CARRIES ITS LEG COUNT AND ITS RATE
#   : inside the bar, or above the bars with the arm name where a group's bars are too short for the label.
#   Also written, for comparison only and not cited: the line version with shaded intervals (APPROACH_FUNNEL_BY_ARM_LINES).
#   Style, fonts and note strip from figure_style.py, as the v2 charts. No 35% floor (it is defined on the pooled capture rate).
# SOURCES (nothing typed): analysis/results_tables/RESULTS_TABLES.md table T14 (cells with the seed-cluster bootstrap).
# VALIDATION, ENFORCED (refuses): every count against APPROACH_FUNNEL.csv (truth, nine wind stages), 84 cells; PNG width.
# CLI: (none) SIDE-EFFECTS: writes APPROACH_FUNNEL_BY_ARM.png/.pdf/.md under OUT/figures.
# =============================================================================
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import os, re, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figure_style as S  # noqa: E402
plt = S.plt
STEM = "APPROACH_FUNNEL_BY_ARM"
TABLES = (_REPO + "/analysis/results_tables/RESULTS_TABLES.md")
SRC = os.path.join(S.OUT, "APPROACH_FUNNEL.csv")
S.FONT, S.FONT_ROW = S.FONT + 0.5, S.FONT_ROW + 0.5
plt.rcParams.update({"font.size": S.FONT, "axes.labelsize": S.FONT, "axes.titlesize": S.FONT, "xtick.labelsize": S.FONT, "ytick.labelsize": S.FONT})
STEPS = [("within 10 m of the tip", "≤ 10 m"), ("within 3 m", "≤ 3 m"), ("within 1.5 m", "≤ 1.5 m"), ("inside the M3b box", "In M3b"),
         ("in the box >= 1 s cumulative", "In M3b\n≥ 1 s"), ("in the box >= 3 s cumulative", "In M3b\n≥ 3 s"), ("M3b success (flag of record)", "Capture")]
ARM_SRC = {"S1O": "FlightDock-S1O", "S1": "FlightDock-S1", "A2O": "FlightDock-A2O", "A2": "FlightDock-A2", "PIDO": "PIDO", "PID": "PID-baseline (comparator)"}
CELL = re.compile(r"(\d+) of (\d+), ([\d.]+)% \[([\d.]+), ([\d.]+)\]")
TITLE = "Approach Funnel by Arm"
NOTE = ("Each point is the share of an arm's 540 wind-stage legs that reached the step at least once, truth channel: within 10 m, 3 m and 1.5 m of the boom-tip; inside the M3b envelope; "
        "inside it for 1 s and for 3 s in all (cumulative, not continuous); and a capture under the four conditions of the document. Shaded band = two-sided 95% seed-cluster bootstrap "
        "interval. Solid line and hollow marker = with the 0.25 m offset (S1O, A2O, PIDO); dotted line and filled marker = without it (S1, A2, PID). Left: light turbulence (σ = 0.25); "
        "right: moderate turbulence (σ = 0.35). Nine wind stages pooled. Amplifying; decides no hypothesis.")
NOTE_IN = 0.7


def load():
    txt = open(TABLES, encoding="utf-8").read(); m = re.search(r"^### T14\. .*?(?=^### )", txt, flags=re.S | re.M)
    if not m:
        raise SystemExit("REFUSE: table T14 not found")
    src = pd.read_csv(SRC); src = src[src.pool == "nine wind stages"]; cells, n = {}, 0
    for line in m.group(0).splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) != 9 or c[0] in ("σ", "") or set(c[0]) <= set("-"):
            continue
        sig, arm = float(c[0]), c[1]
        for (step, _), cell in zip(STEPS, c[2:]):
            g = CELL.match(cell)
            if not g:
                raise SystemExit("REFUSE: cannot parse %r" % cell)
            k, tot, pct, lo, hi = int(g.group(1)), int(g.group(2)), float(g.group(3)), float(g.group(4)), float(g.group(5))
            q = src[(src.arm == ARM_SRC[arm]) & (src.sigma == sig) & (src.step == step)]
            if len(q) != 1 or not str(q.truth.iloc[0]).startswith("%d of %d," % (k, tot)) or tot != 540:
                raise SystemExit("REFUSE: %s σ %.2f %s does not reproduce APPROACH_FUNNEL.csv" % (arm, sig, step))
            cells[(sig, arm, step)] = (pct, lo, hi); n += 1
    if n != 84:
        raise SystemExit("REFUSE: %d cells, expected 84" % n)
    print("VALIDATION PASS: 84 funnel cells reproduce APPROACH_FUNNEL.csv")
    return cells


COUNT = re.compile(r"(\d+) of 540")
NOTE_COLS = ("Each bar is the share of an arm's 540 wind-stage legs that reached the step at least once, truth channel: within 10 m, 3 m and 1.5 m of the boom-tip; inside the M3b envelope; "
             "inside it for 1 s and for 3 s in all (cumulative, not continuous); and a capture under the four conditions of the document. Whisker = two-sided 95% seed-cluster bootstrap interval; "
             "the label is the leg count and the rate (count ÷ 540), on the bar, or above the bars with the arm name where they are too short for it. "
             "In each group the bars run S1O, S1, A2O, A2, PIDO, PID. Dark bars = with the 0.25 m offset (S1O, A2O, PIDO); light bars = without it "
             "(S1, A2, PID). Top: light turbulence (σ = 0.25); bottom: moderate turbulence (σ = 0.35). Nine wind stages pooled. Amplifying; decides no hypothesis.")


def counts():
    txt = open(TABLES, encoding="utf-8").read(); m = re.search(r"^### T14\. .*?(?=^### )", txt, flags=re.S | re.M); out = {}
    for line in m.group(0).splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) == 9 and c[0] in ("0.25", "0.35"):
            for (step, _), cell in zip(STEPS, c[2:]):
                out[(float(c[0]), c[1], step)] = int(COUNT.match(cell).group(1))
    return out


def _span(ax, t, r):
    """Vertical extent of a text artist in data units of its axes."""
    bb = t.get_window_extent(r); inv = ax.transData.inverted()
    return inv.transform((0, bb.y0))[1], inv.transform((0, bb.y1))[1]


def columns(cells):
    """The operator's earlier funnel figure: bars, one panel per turbulence level, whisker = CI. Column style of H1_CORRIDOR_P95_MARGIN_v2.
    Every bar carries its leg count and its rate. The label sits inside the bar, under the whisker, at one height per
    group; where any bar of a group is too short for it, the whole group carries arm name, count and rate above the bars. Refuses on a rate that is not count / 540,
    on a label that leaves the axes, and on a label that crosses a whisker."""
    import matplotlib.patheffects as pe
    k_of = counts(); n_arm = len(S.ROW_ORDER); bw = 0.84 / n_arm; groups = []
    fig = plt.figure(figsize=(S.TEXT_W_LANDSCAPE, 2 * 2.55 + 0.75)); gs = fig.add_gridspec(3, 1, height_ratios=[2.55, 2.55, 0.75]); fig.set_layout_engine("constrained")
    for i, sig in enumerate((0.25, 0.35)):
        ax = fig.add_subplot(gs[i, 0])
        for j, (step, _) in enumerate(STEPS):
            grp = []
            for a, k in enumerate(S.ROW_ORDER):
                pct, lo, hi = cells[(sig, k, step)]; x = j - 0.42 + bw * (a + 0.5); cnt = k_of[(sig, k, step)]
                if abs(100.0 * cnt / 540 - pct) > 0.05:
                    raise SystemExit("REFUSE: %s σ %.2f %s: the rate %.1f%% is not %d of 540" % (k, sig, step, pct, cnt))
                ax.bar(x, pct, width=bw * 0.9, facecolor=S.COLOUR[k], edgecolor=S.DARK[k], linewidth=0.9, zorder=3)
                ax.errorbar([x], [pct], yerr=[[pct - lo], [hi - pct]], fmt="none", ecolor="k", elinewidth=0.8, capsize=1.8, capthick=0.8, zorder=5)
                t = ax.text(x, lo * 0.5, "%d (%.1f%%)" % (cnt, pct), rotation=90, ha="center", va="center", color="white", fontsize=S.FONT_ROW, fontweight="bold", zorder=6,
                            path_effects=[pe.withStroke(linewidth=1.1, foreground=S.DARK[k])])
                grp.append((k, x, lo, hi, t))
            groups.append((ax, j, grp))
        ax.set_xticks(range(len(STEPS))); ax.set_xticklabels([lab.replace("\n", " ") for _, lab in STEPS]); ax.set_xlim(-0.55, len(STEPS) - 0.45)
        ax.set_ylim(0, 108); ax.set_yticks(range(0, 101, 20)); ax.grid(axis="y", alpha=0.3); ax.set_ylabel("Legs reaching the step (% of 540)")
        ax.set_title("%s turbulence (σ = %.2f)" % ("Light" if sig == 0.25 else "Moderate", sig), loc="left", fontsize=S.FONT)
    S.crop_note(fig, fig.add_subplot(gs[2, 0]), 'Approach Funnel by Arm. Note. Each bar is the share of an arm’s 540 wind-stage legs that reached the step at least once, truth channel: within 10 m, 3 m and 1.5 m of the boom-tip; inside the M3b envelope; inside it for 1 s and for 3 s in all (cumulative, not continuous); and a capture under the four conditions of the “Decision Rules” section. Whisker = two-sided 95% seed-cluster bootstrap interval; the label is the leg count and the rate (count ÷ 540), on the bar, or above the bars with the arm name where they are too short for it. In each group the bars run S1O, S1, A2O, A2, PIDO, PID. Dark bars = with the 0.25 m offset (S1O, A2O, PIDO); light bars = without it (S1, A2, PID). Top: light turbulence (σ = 0.25); bottom: moderate turbulence (σ = 0.35). Nine wind stages pooled. Amplifying; decides no hypothesis.', S.TEXT_W_LANDSCAPE)
    fig.canvas.draw(); r = fig.canvas.get_renderer(); n_in = n_above = 0
    for ax, j, grp in groups:
        need = max(y1 - y0 for y0, y1 in (_span(ax, t, r) for *_, t in grp)); room = min(lo for _, _, lo, _, _ in grp)
        if need <= room - 2.0:   # every label of the group fits between the axis and the lowest whisker end: one common height for the group
            for k, x, lo, hi, t in grp:
                t.set_position((x, room * 0.5)); n_in += 1
                if j == len(STEPS) - 1:   # arm names above the bars of the last group, where there is headroom
                    ax.text(x, hi + 2.5, S.SHORT[k], rotation=90, ha="center", va="bottom", color=S.DARK[k], fontsize=S.FONT_ROW, fontweight="bold", zorder=6)
        else:                    # too short for the label: arm name, count and rate above the bars of the whole group
            for k, x, lo, hi, t in grp:
                t.set_text("%s %s" % (S.SHORT[k], t.get_text())); t.set_position((x, hi + 2.5)); t.set_va("bottom"); t.set_color(S.DARK[k]); t.set_path_effects([]); n_above += 1
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for ax, j, grp in groups:
        for k, x, lo, hi, t in grp:
            y0, y1 = _span(ax, t, r)
            if y0 < 0.5 or y1 > ax.get_ylim()[1] - 0.5:
                raise SystemExit("REFUSE: the label %r leaves the axes (%.1f to %.1f)" % (t.get_text(), y0, y1))
            if y0 < hi and y1 > lo:
                raise SystemExit("REFUSE: the label %r crosses its whisker (%.1f to %.1f against %.1f to %.1f)" % (t.get_text(), y0, y1, lo, hi))
    if n_in + n_above != 84 or n_above == 0:
        raise SystemExit("REFUSE: %d labels inside and %d above; the note says some sit above the bars and 84 are expected" % (n_in, n_above))
    print("bar labels (count and rate): %d inside the bar, %d above the bars with the arm name" % (n_in, n_above))
    S.save_exact(fig, STEM, S.TEXT_W_LANDSCAPE)
    open(os.path.join(S.FIGDIR, STEM + ".md"), "w").write("# Approach funnel by arm (figure, bars) - POST-PLAN PRESENTATION; decides nothing\n\nTitle: " + TITLE + "\n\nFigure note: " + NOTE_COLS + "\n")


def lines(cells, y0=15):
    """The line version with shaded intervals, y axis from y0; written beside the bars for comparison."""
    fig = plt.figure(figsize=(S.TEXT_W_LANDSCAPE, 3.6 + NOTE_IN)); gs = fig.add_gridspec(2, 2, height_ratios=[3.6, NOTE_IN]); fig.set_layout_engine("constrained")
    X = list(range(len(STEPS)))
    for i, sig in enumerate((0.25, 0.35)):
        ax = fig.add_subplot(gs[0, i])
        for k in S.ROW_ORDER:
            v = [cells[(sig, k, s)] for s, _ in STEPS]
            ax.fill_between(X, [a[1] for a in v], [a[2] for a in v], color=S.COLOUR[k], alpha=0.18, lw=0, zorder=2)
            ax.plot(X, [a[0] for a in v], color=S.COLOUR[k], ls=S.LINESTYLE[k], lw=S.LW_LINE * 0.6, marker=S.MARKER[k], ms=S.MS * 0.6, mfc=S.marker_face(k), mec=S.DARK[k], mew=0.9, zorder=3)
        ends = sorted(((cells[(sig, k, STEPS[-1][0])][0], k) for k in S.ROW_ORDER)); pos, y_prev = [], -1e9
        for y, k in ends:
            yy = max(y, y_prev + 3.4); y_prev = yy; pos.append(yy)
        shift = sum(y for y, _ in ends) / len(ends) - sum(pos) / len(pos)
        for (y, k), yy in zip(ends, pos):
            ax.text(X[-1] + 0.18, yy + shift, S.SHORT[k], color=S.DARK[k], fontweight="bold", fontsize=S.FONT_ROW, va="center", ha="left")
        ax.set_xticks(X); ax.set_xticklabels([lab for _, lab in STEPS]); ax.set_xlim(-0.3, X[-1] + 0.9); ax.set_ylim(y0, 102)
        ax.set_yticks(range(20, 101, 20)); ax.grid(axis="y", alpha=0.3); ax.set_ylabel("Legs reaching the step (% of 540)" if i == 0 else "")
        ax.set_title("%s turbulence (σ = %.2f)" % ("Light" if sig == 0.25 else "Moderate", sig), loc="left", fontsize=S.FONT)
    S.crop_note(fig, fig.add_subplot(gs[1, :]), "Title: " + TITLE + " || Note: " + NOTE, S.TEXT_W_LANDSCAPE)
    S.save_exact(fig, STEM + "_LINES", S.TEXT_W_LANDSCAPE)


def main():
    cells = load(); columns(cells); lines(cells); return 0


if __name__ == "__main__":
    sys.exit(main())
