#!/usr/bin/env python3
# =============================================================================

# 2026-10-02: MOVED TO THE APPENDIX (provisionally ); the chapter's the figure “H2: Capture Rate Comparison, Light Turbulence” is now h2_pooled_wind_margin_v2.py. Nothing drawn changed.
#   REBUILT from the 09-19 version (archive_captures_wilson_20261001/): palette of record, print size, crop note, slide mode,
#   and EVERY interval is the two-sided 95% seed-cluster bootstrap ("For every table and figure we make,
#   we MUST include seed-cluster bootstrap"). Wilson intervals are gone from the figure and the tables.
#   POST-PLAN PRESENTATION; decides nothing.
# WHAT IS DRAWN: docking-envelope captures (the flag of record, board `passed`) as a share of the 60 legs flown per stage and
#   arm, one line per arm over the nine wind stages, sigma 0.25 panel above sigma 0.35, Calm Control as a lone marker with its
#   whisker left of 0 m/s (it is not a wind stage and enters no pool). 10-01 revisions: no strong-wind shading, y ticks every 10%,
#   tick row = mean wind 0.0 ... 8.0 with the row header "Mean Wind (m/s)", y values on both panels, caps on the Calm whisker.
#   Band = two-sided 95% seed-cluster bootstrap interval of each stage rate (30 seeds resampled whole, 10,000 draws, fixed
#   seed, p95_boot._picks: the same draws as every interval of record). Arm names at the right end of each line (no legend).
#   The tables add the strong-wind pool (180 legs) and the nine-stage pool (540 legs) with the same interval.
# SOURCES (nothing typed): board/BOARD_W10.csv via amplifying_tables.arm_sigma; p95_boot for the draws.
# VALIDATION, ENFORCED (refuses): per-stage capture and leg counts == comparative/SAFETY_BY_STAGE.csv; the sigma 0.25 pooled
#   capture counts of every arm (nine stages, 540; strong wind, 180) == figures/H2_POOLED_WIND_MARGIN.csv (arm and PID
#   columns); seed block 10775-10804, 30 seeds in every cell, 60 legs per stage; every interval brackets its rate; PNG width.
# CLI: --side-by-side draws the two panels LEFT (σ 0.25) and RIGHT
#   (σ 0.35) at the landscape print width (9.0 in) as DOCKING_ENVELOPE_CAPTURES_BY_WIND_STAGE_SIDE_BY_SIDE; the tables are the same.
# SIDE-EFFECTS: writes DOCKING_ENVELOPE_CAPTURES_BY_WIND_STAGE[_SIDE_BY_SIDE].png/.pdf/.md and _SIGMA_<025|035>.csv/.md.
# =============================================================================
# 2026-10-01 ADVISOR RULING: POOLED WINDS ONLY. The strong-wind pool, strong_wind_pools_appendix.py and their
#   outputs are ARCHIVED in archive_strong_wind_20261001/; older comments below that mention the figure are historical.
#   WORDING: "comparator (not graded)" replaces "reported, not graded" for the PID-baseline and PIDO; the PID H2 row reads
#   "baseline comparator (not graded)"; sigma 0.35 and other ungraded readings say "not graded for H1".
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figure_style as S  # noqa: E402
import amplifying_tables as AT  # noqa: E402
import p95_boot as PB  # noqa: E402

plt = S.plt
STEM = "DOCKING_ENVELOPE_CAPTURES_BY_WIND_STAGE"
STEMF = STEM + "_SIGMA_%s"
STRONG, NINE, CALM = "strong wind, 6-8 m/s", "nine wind stages", "S0 calm control stage"
STAGES = [CALM] + ["%d m/s" % k for k in range(9)]
GROUPS = [NINE] + STAGES   # 2026-10-01: pooled winds only; the strong-wind pool is archived
ROWS = list(S.ROW_ORDER)
X = np.array([-1.2] + list(range(9)), float)
FLOOR = 35.0   # minimum capture rate of reference: manned aerial-refueling docking success, Ren and Quan (2024); reported, not a hypothesis
CALM_DX = {k: dx for k, dx in zip(S.ROW_ORDER, np.linspace(-0.3, 0.3, 6))}   # the six Calm Control markers side by side
SIDE = "--side-by-side" in sys.argv
PANEL = {0.25: "Light turbulence (σ 0.25)", 0.35: "Moderate turbulence (σ 0.35)"}   # package 10 (2026-10-02): plain panel titles in every mode
DX = {k: dx for k, dx in zip(S.ROW_ORDER, np.linspace(-0.22, 0.22, 6))}   # package 10 option A: the six arms dodged along the wind axis so the whiskers stay apart
TITLE = "Docking-Envelope Captures by Wind Stage and Arm, Light (σ = 0.25) and Moderate (σ = 0.35) Turbulence"
DESC = ("Share of the 60 legs flown per stage and arm that captured the docking envelope (the four-condition flag of record), 30 seeds "
        "times two trajectories per stage, seed block 10775-10804. Lines join the stage rates; the thin gray capped whisker at each point is the "
        "two-sided 95% seed-cluster bootstrap interval of that rate (30 seeds resampled whole, 10,000 draws), and the arms are offset slightly along the wind axis so the whiskers stay apart. Stage 0 is Calm Control, "
        "no wind and no turbulence, and enters no pool; the two Calm Control points are the same 60 legs flown twice, once at the start "
        "of each ladder, and the gap between them is the run-to-run repeatability noise reported in the table “Truth vs. Perception: Corridor Error by Axis”. The nine wind stages pooled at "
        "σ 0.25 are the H2 pool; σ 0.35 is the moderate level of the H3 comparison. The dotted gray line at 35% is the minimum capture rate of reference, the manned aerial-refueling docking success rate reported by Ren and Quan (2024); it applies to the pooled nine-stage rate and decides nothing. Pooled rates are in the table “Calm Control Repeatability”. Higher is better.")
FIGURE_NOTE = ("Solid lines and hollow markers = the arms with the 0.25 m offset (S1O, A2O, PIDO); dotted lines and filled markers = the same "
               "controllers without it (S1, A2, PID). Thin gray capped whisker at each point (and the colored whisker at Calm Control) = two-sided 95% seed-cluster bootstrap "
               "interval of each stage rate. Dotted gray line = 35% reference floor (pooled rate; reported, not a hypothesis).")
PCTCI = "rate [two-sided 95% seed-cluster bootstrap]"


def truthy(s):
    return s.astype(str).str.lower().eq("true")


def boot_rate(ok, seed):
    """Two-sided 95% seed-cluster bootstrap interval (in %) of a capture rate: whole seeds, PB's draws, ratio of resampled sums."""
    labels = np.sort(np.unique(seed))
    if list(labels) != S.SEED_BLOCK:
        raise SystemExit("REFUSE: a cell does not hold the 30 seeds of block 10775-10804")
    k = np.array([ok[seed == s].sum() for s in labels], dtype=float)
    n = np.array([(seed == s).sum() for s in labels], dtype=float)
    picks = PB._picks(len(labels))
    vals = 100.0 * k[picks].sum(axis=1) / n[picks].sum(axis=1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cell(d, keys, label):
    rows = []
    for key, g in d.groupby(keys, sort=False):
        ok, seed = g.ok.to_numpy(), g.seed.to_numpy()
        lo, hi = boot_rate(ok, seed)
        rows.append(dict(zip(keys, key), k=int(ok.sum()), n=int(len(ok)), pct=100.0 * ok.mean(), lo=lo, hi=hi, stage=label if label else key[-1]))
    return pd.DataFrame(rows)


def load():
    board = pd.read_csv(os.path.join(S.RUN, "board", "BOARD_W10.csv"), low_memory=False)
    b = AT.arm_sigma(board)
    b["ok"], b["wind"], b["strong"] = truthy(b.passed), truthy(b.wind_rung), truthy(b.sw)
    for c, g in b.groupby("corpus"):
        if sorted(g.seed.unique()) != S.SEED_BLOCK or not (g.groupby("wind_label").size() == 60).all():
            raise SystemExit("REFUSE: corpus %s is not seed block 10775-10804 with 60 legs per stage" % c)
    per = cell(b, ["arm", "sig", "stage"], None)
    sb = pd.read_csv(os.path.join(S.RUN, "comparative", "SAFETY_BY_STAGE.csv")).rename(columns={"sigma": "sig"})
    m = per.merge(sb[["arm", "sig", "stage", "n", "captured"]], on=["arm", "sig", "stage"], how="outer", validate="one_to_one", indicator=True)
    if not (m._merge == "both").all() or (m.k != m.captured).any() or (m.n_x != m.n_y).any():
        raise SystemExit("REFUSE: per-stage capture counts do not reproduce SAFETY_BY_STAGE.csv")
    d = pd.concat([per, cell(b[b.wind], ["arm", "sig"], NINE)], ignore_index=True)
    d["key"] = d.arm.map({S.V.NAME[k]: k for k in S.ARMS})
    if d.key.isna().any() or not ((d.lo <= d.pct) & (d.pct <= d.hi)).all():
        raise SystemExit("REFUSE: unknown arm name or an interval that does not bracket its rate")
    h2 = pd.read_csv(os.path.join(S.FIGDIR, "H2_POOLED_WIND_MARGIN.csv"))
    h2["pool"] = NINE   # H2_POOLED_WIND_MARGIN.csv carries the nine-stage pool only (strong wind archived 2026-10-01)
    count = lambda s: int(re.match(r"(\d+) \(", s).group(1))
    checks = 0
    for _, r in h2[h2.arm != "PID"].iterrows():
        for k, col in ((r.arm, "docking-envelope captures, arm"), ("PID", "docking-envelope captures, PID-baseline (comparator)")):
            x = d[(d.key == k) & (d.sig == 0.25) & (d.stage == r.pool)]
            if len(x) != 1 or int(x.k.iloc[0]) != count(r[col]) or int(x.n.iloc[0]) != int(r["same-seed pairs"]):
                raise SystemExit("REFUSE: %s %s pooled captures differ from H2_POOLED_WIND_MARGIN.csv" % (k, r.pool))
            checks += 1
    print("VALIDATION PASS: %d stage cells == SAFETY_BY_STAGE.csv; %d pooled counts == H2_POOLED_WIND_MARGIN.csv; 30 seeds in every cell; "
          "intervals: seed-cluster bootstrap, %d draws, seed %d" % (len(per), checks, PB.BOOT, PB.SEED))
    return d


def end_labels(ax, ys, min_gap):
    """Arm names at the right end of each line, pushed apart so no two overlap (data units, bottom-up)."""
    order = sorted(ys, key=lambda k: ys[k])
    pos, last = {}, -1e9
    for k in order:
        pos[k] = max(ys[k], last + min_gap)
        last = pos[k]
    shift = (pos[order[-1]] - ys[order[-1]]) / 2.0 if pos[order[-1]] > ax.get_ylim()[1] - min_gap else 0.0   # keep inside the axes
    for k in order:
        if k == "__floor__":   # package 10: the floor label is dodged with the arm names so nothing collides
            ax.text(X[-1] + 0.45, pos[k] - shift, "35% floor", color="0.3", fontsize=S.FONT_ROW, va="center", ha="left", zorder=6, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
            continue
        ax.text(X[-1] + 0.45, pos[k] - shift, S.SHORT[k], color=S.DARK[k], fontsize=S.FONT_ROW + 0.5, va="center", ha="left",
                fontweight="bold" if k in S.HOLLOW else "normal", zorder=6)


def draw_panel(fig, ax, d, sig, last, ylabel=True):
    ends = {}
    for k in ROWS:
        g = d[(d.key == k) & (d.sig == sig)].set_index("stage").loc[STAGES]
        mk = dict(marker=S.MARKER[k], ms=S.MS * 0.6, mfc=S.marker_face(k), mec=S.DARK[k], mew=0.9, zorder=3)
        xs = X[1:] + DX[k]   # package 10 (option A): thin gray capped whisker at each point = the seed-cluster bootstrap interval of that stage rate; the bands are gone
        ax.errorbar(xs, g.pct.iloc[1:].to_numpy(), yerr=[(g.pct.iloc[1:] - g.lo.iloc[1:]).to_numpy(), (g.hi.iloc[1:] - g.pct.iloc[1:]).to_numpy()], fmt="none", ecolor="0.45", elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=2)
        ax.plot(xs, g.pct.iloc[1:].to_numpy(), color=S.COLOUR[k], ls=S.LINESTYLE[k], lw=S.LW_LINE * 0.6, **mk)
        ax.errorbar([X[0] + CALM_DX[k]], [g.pct.iloc[0]], yerr=[[g.pct.iloc[0] - g.lo.iloc[0]], [g.hi.iloc[0] - g.pct.iloc[0]]],   # Calm Control stands alone:
                    color=S.COLOUR[k], elinewidth=S.LW_LINE * 0.45, capsize=2.2, capthick=S.LW_LINE * 0.45, ls="none", **mk)   # caps: operator 10-01 # no pool, not a wind stage
        ends[k] = float(g.pct.iloc[-1])
    ax.set_xticks(X)
    ax.set_xticklabels(["Calm"] + ["%.1f" % k for k in range(9)])   # operator 10-01: wind row only, 0.0 ... 8.0; no stage row
    ax.set_xlim(-1.85, 10.25)
    ax.axhline(FLOOR, color="0.3", ls=(0, (1, 2)), lw=1.0, zorder=2)
    ax.set_ylim(0, 102)
    ax.set_yticks(range(0, 101, 10))   # operator 10-01: 10% steps
    ax.grid(alpha=0.3)
    if ylabel:
        ax.set_ylabel("docking-envelope captures\n(% of 60 legs per stage)")
    ax.tick_params(axis="x", labelbottom=last)
    if last:   # row header left of the tick row (operator 10-01: "Mean Wind (m/s)" on the left side for the wind row)
        import matplotlib.transforms as mtransforms
        ax.text(-0.012, 0, "Mean Wind (m/s)", ha="right", va="top", fontsize=S.FONT,
                transform=mtransforms.blended_transform_factory(ax.transAxes, ax.transAxes) + mtransforms.ScaledTranslation(0, -4.5 / 72.0, fig.dpi_scale_trans))
    ends["__floor__"] = FLOOR
    end_labels(ax, ends, 5.0)
    S.fit_title(fig, ax, PANEL[sig])


NOTE_IN = 1.2   # inches reserved under the chart for the crop note
PANEL_IN = 2.15


def figure(d):
    if SIDE:   # landscape: σ 0.25 left, σ 0.35 right, shared y, one note row under both
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(S.TEXT_W_LANDSCAPE, 3.3 + NOTE_IN * 0.75))
        gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[3.3, NOTE_IN * 0.75])
        axl = fig.add_subplot(gs[0, 0]); axr = fig.add_subplot(gs[0, 1], sharey=axl); axn = fig.add_subplot(gs[1, :])
        fig.set_layout_engine("constrained")
        draw_panel(fig, axl, d, 0.25, True)
        draw_panel(fig, axr, d, 0.35, True, ylabel=False)   # y values on both panels
        S.crop_note(fig, axn, 'Captures by Wind Stage and Arm. Note. Capture = truth and perception both inside M3b for 3.0 s (the “Decision Rules” section). Solid lines and hollow markers = the arms with the 0.25 m offset (S1O, A2O, PIDO); dotted lines and filled markers = the same controllers without it (S1, A2, PID). Thin gray capped whisker at each point (and the colored whisker at Calm Control) = two-sided 95% seed-cluster bootstrap interval of each stage rate. Dotted gray line = 35% reference floor (pooled rate; reported, not a hypothesis).', S.TEXT_W_LANDSCAPE)
        S.save_exact(fig, STEM + "_SIDE_BY_SIDE", S.TEXT_W_LANDSCAPE)
        return
    fig, axs = plt.subplots(3, 1, figsize=(S.TEXT_W_PORTRAIT, 2 * PANEL_IN + NOTE_IN), gridspec_kw={"height_ratios": [PANEL_IN, PANEL_IN, NOTE_IN], "hspace": 0.02})
    fig.set_layout_engine("constrained")
    draw_panel(fig, axs[0], d, 0.25, False)
    draw_panel(fig, axs[1], d, 0.35, True)
    S.crop_note(fig, axs[2], "Title: " + TITLE + " || Description: " + DESC + " || Legend: " + FIGURE_NOTE, S.TEXT_W_PORTRAIT)
    S.save_exact(fig, STEM, S.TEXT_W_PORTRAIT)


def table(d, sig):
    rows = []
    for s in GROUPS:
        pid = float(d[(d.key == "PID") & (d.stage == s)].pct.iloc[0])
        for k in ROWS:
            r = d[(d.key == k) & (d.stage == s)].iloc[0]
            rows.append({"stage": {NINE: "All nine wind stages" + (" (H2 pool)" if sig == 0.25 else ""), CALM: "Calm Control"}.get(s, s),
                         "arm": S.SHORT[k], "docking-envelope captures": "%d of %d" % (r.k, r.n), PCTCI: "%.1f%% [%.1f, %.1f]" % (r.pct, r.lo, r.hi),
                         "rate minus PID-baseline rate (pp), unpaired": "%+.1f" % (r.pct - pid)})
    t = pd.DataFrame(rows)
    stem = os.path.join(S.FIGDIR, STEMF % ("%.2f" % sig).replace(".", ""))
    t.to_csv(stem + ".csv", index=False)
    notes = ["Docking-envelope captures = the flag of record (board `passed`), as a share of legs flown. Sigma %.2f; seed block 10775-10804; 30 seeds x 2 trajectories per stage." % sig,
             "Intervals: two-sided 95%% seed-cluster bootstrap (30 seeds resampled whole, %d draws, fixed seed %d, percentile method), the same draws as every interval of record. "
             "The difference column is a plain subtraction of rates and decides nothing; the H2 statistic is the PAIRED nine-stage difference at sigma 0.25 in H2_POOLED_WIND_MARGIN." % (PB.BOOT, PB.SEED)]
    open(stem + ".md", "w").write("# Docking-envelope captures by wind stage, sigma %.2f - POST-PLAN PRESENTATION; decides nothing\n\n" % sig
                                  + S.md(t) + "\n\n" + "\n".join("- " + x for x in notes) + "\n")
    print("wrote", stem + ".csv|.md")


def figure_md():
    p = os.path.join(S.FIGDIR, STEM + ("_SIDE_BY_SIDE" if SIDE else "") + ".md")
    open(p, "w").write("# Docking-envelope captures by wind stage - POST-PLAN PRESENTATION; decides nothing\n\n"
                       "Suggested figure title: " + TITLE + "\n\nFigure description (for the document, not drawn in the figure): " + DESC +
                       "\n\nFigure note (the legend, for your 9 pt note under the figure): " + FIGURE_NOTE +
                       "\n\nPalette: " + S.PALETTE_RULE + ".\n\nTables: " + STEMF % "025" + ".csv|.md and " + STEMF % "035" + ".csv|.md.\n")
    print("wrote", p)


def main():
    d = load()
    for sg in (0.25, 0.35):
        x = d[d.sig == sg]
        if len(x) != len(ROWS) * len(GROUPS):
            raise SystemExit("REFUSE: sigma %.2f does not have one cell per arm and group" % sg)
        table(x, sg)
    figure(d)
    figure_md()
    return 0


if __name__ == "__main__":
    sys.exit(main())
