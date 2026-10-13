#!/usr/bin/env python3
# =============================================================================

#   racetrack, drawn with the panel code of the v2 charts of record (H1_CORRIDOR_P95_MARGIN_v2, H2_POOLED_WIND_MARGIN_v2,
#   H3_POOLED_WIND_DECREASE_v2). POST-PLAN PRESENTATION; decides nothing: no trajectory panel is graded (every row reads
#   "not graded"); the hypotheses are graded on the nine-stage pool of both trajectories only.
# WHAT IS DRAWN: per trajectory, nine wind stages pooled, 270 legs / same-seed pairs per arm; the same statistics, the same
#   two-sided 95% seed-cluster bootstrap (30 seeds resampled whole, the draws of record) as the pooled charts.
# VALIDATION, ENFORCED (refuses): straight + racetrack counts must sum to the pooled counts of record recomputed here with the
#   v2 loaders (which themselves refuse unless the files of record reproduce); the p95 per trajectory must reproduce
#   OFFSET_EFFECT_BY_OUTCOME.csv (twelve values at sigma 0.25, 3 dp); PNG width; label overlap (the v2 checks).
# CLI: (none) SIDE-EFFECTS: writes H1_/H2_/H3_BY_TRAJECTORY.png/.pdf and HYPOTHESES_BY_TRAJECTORY.csv/.md under OUT/figures.
# =============================================================================
import os as _os; _REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../.."))
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figure_style as S  # noqa: E402
import h1_corridor_p95_margin_v2 as M1  # noqa: E402
import h2_pooled_wind_margin_v2 as M2  # noqa: E402
import h3_pooled_wind_decrease_v2 as M3  # noqa: E402
plt, V, HA, AT, PB = S.plt, S.V, M2.HA, M1.AT, M3.PB
TRAJ = ["straight", "racetrack"]
OFF = (_REPO + "/analysis/offset_effect_20260920/OFFSET_EFFECT_BY_OUTCOME.csv")
NG = "not graded"
NOTE_TAIL = " Each panel holds one DHC-2 trajectory, nine wind stages pooled, 270 legs or same-seed pairs per arm. No trajectory panel is graded; the hypotheses are graded on both trajectories pooled. Amplifying; decides no hypothesis."
rows_out = []


def refuse(m):
    raise SystemExit("REFUSE: " + m)


def h1_cells():
    pooled, _ = M1.load()
    board = pd.read_csv(os.path.join(S.RUN, "board", "BOARD_W10.csv"), low_memory=False)
    b = AT.arm_sigma(board).merge(pd.read_csv(os.path.join(S.OUT, "PERCEPTION_PEAKS_PER_LEG.csv")), on=["corpus", "leg"], how="left", validate="one_to_one")
    b = b[b.wind_rung.astype(str).str.lower().eq("true")]
    off = pd.read_csv(OFF); off = off[(off.pool == "nine wind stages") & (off.sigma == 0.25) & (off.outcome == "corridor p95 (m)")]
    fam = {"S1": ("FlightDock-S1", "offset_off"), "S1O": ("FlightDock-S1", "offset_on"), "A2": ("FlightDock-A2", "offset_off"), "A2O": ("FlightDock-A2", "offset_on"),
           "PID": ("PID-baseline", "offset_off"), "PIDO": ("PID-baseline", "offset_on")}
    out = {}
    for t in TRAJ:
        cells = {}
        for k in M1.ROWS:
            x = b[(b.corpus == V.corpus(k, 0.25)) & (b.trajectory == t)]
            if len(x) != 270: refuse("H1 %s %s has %d legs" % (k, t, len(x)))
            p, lo, hi = HA.pct_ci(x, [95]); p, lo, hi = round(float(p[0]), 3), round(float(lo[0]), 3), round(float(hi[0]), 3)
            q = off[off.controller.str.startswith(fam[k][0]) & (off.trajectory == t)]
            if len(q) != 1 or abs(float(q[fam[k][1]].iloc[0]) - p) > 0.0006: refuse("H1 %s %s p95 %.3f does not reproduce OFFSET_EFFECT_BY_OUTCOME.csv" % (k, t, p))
            xp = x[["seed", "perception_corridor"]].rename(columns={"perception_corridor": "win"}); pp, plo, phi = HA.pct_ci(xp, [95])
            cells[k] = dict(p95=p, two=(lo, hi), perc=round(float(pp[0]), 3), perc_two=(round(float(plo[0]), 3), round(float(phi[0]), 3)), n=int(x.win.notna().sum()), p_one=None)
            rows_out.append({"chart": "H1 corridor p95 (m)", "trajectory": t, "arm": k, "n": cells[k]["n"], "value": p, "ci_lo": lo, "ci_hi": hi})
        out[t] = cells
    for k in M1.ROWS:
        if out["straight"][k]["n"] + out["racetrack"][k]["n"] != pooled[k]["n"]: refuse("H1 %s corridor legs do not sum to the pooled count" % k)
    return out


def paired_cells(mod, kind):
    pooled, _ = mod.load(); bd = HA.P.board(); out = {}
    for t in TRAJ:
        for k in mod.ROWS:
            mask = lambda d, t=t: d["wind_rung"] & (d.trajectory == t)
            if kind == "H2":
                if k == "PID":
                    out[(k, t)] = dict(n=270, ka=None, kb=None, diff=0.0, b=None, c=None, two=(None, None), p_one=None); continue
                a = V.corpus(k, 0.25); r = V.paired(bd, a, V.corpus("PID", 0.25), mask)
            else:
                a = V.corpus(k, 0.25); r = V.paired(bd, a, V.corpus(k, 0.35), mask)
            if r["n"] != 270: refuse("%s %s %s has %d pairs" % (kind, k, t, r["n"]))
            lo, hi = HA.boot_diff(bd, r, a)
            out[(k, t)] = dict(r, two=(lo, hi), p_one=None)
            rows_out.append({"chart": "%s paired difference (pp)" % kind, "trajectory": t, "arm": k, "n": r["n"], "value": round(r["diff"], 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2)})
    for k in mod.ROWS:
        if kind == "H2" and k == "PID": continue
        s, r_, p = out[(k, "straight")], out[(k, "racetrack")], pooled[(k, "nine")]
        if (s["ka"] + r_["ka"], s["kb"] + r_["kb"], s["n"] + r_["n"]) != (p["ka"], p["kb"], p["n"]): refuse("%s %s captures by trajectory do not sum to the pooled counts of record" % (kind, k))
    return out


def title(ax, t, extra):
    ax.set_title("%s trajectory, nine wind stages pooled, %s (not graded)" % (t.capitalize(), extra), loc="left", fontsize=S.FONT_ROW + 0.5)


_NO_OVERLAP = M2.no_overlap


def no_floor(fig, labels, what):
    """The 35% reference floor is defined on the nine-stage pool of both trajectories, so it is not drawn per trajectory."""
    for lab in [x for x in labels if x.get_text().startswith("35%")]:
        lab.remove(); labels.remove(lab)
    for ax in fig.axes:
        for ln in list(ax.lines):
            if str(ln.get_color()).lower() == S.ORANGE.lower() and ln.get_marker() in (None, 'None', '', ' '):   # the floor rule only; a marker-only line can carry the same cycle colour
                ln.remove()
    _NO_OVERLAP(fig, labels, what)


H2_NOTE = ("Marker = each arm's same-seed difference in docking-envelope captures from the PID-baseline, σ = 0.25; whisker = its two-sided 95% seed-cluster bootstrap interval. "
           "The rate above each marker is the arm's capture rate on that trajectory; the value below it is the difference; the interval ends are under the whisker caps. "
           "Dotted green line = 0 pp, the PID-baseline. Hollow marker = with the 0.25 m offset; filled = without. The 35% reference floor is defined on both trajectories pooled and is not drawn.")
H3_NOTE = ("Marker = each arm's same-seed decrease in docking-envelope captures from light (σ = 0.25) to moderate (σ = 0.35) turbulence; whisker = its two-sided 95% seed-cluster "
           "bootstrap interval. The rates above each marker are the arm's capture rates on that trajectory at the two levels; the value below it is the decrease. "
           "Dotted green line = 35 pp. Hollow marker = with the 0.25 m offset; filled = without.")


def main():
    M2.no_overlap = no_floor
    # H1: two column panels side by side
    c1 = h1_cells(); words = {k: NG for k in M1.ROWS}
    keep, keep_w = S.FONT_ROW, M1.WHITE_FONT; S.FONT_ROW, M1.WHITE_FONT = keep * 0.8, keep_w * 0.8   # two columns panels share 9.0 in: row text at 0.8x so the labels fit their slots
    fig = plt.figure(figsize=(S.TEXT_W_LANDSCAPE, 3.6 + 0.75)); gs = fig.add_gridspec(2, 2, height_ratios=[3.6, 0.75]); fig.set_layout_engine("constrained")
    for i, t in enumerate(TRAJ):
        ax = fig.add_subplot(gs[0, i]); M1.draw_panel(fig, ax, c1[t], words); title(ax, t, "σ = 0.25")
    S.crop_note(fig, fig.add_subplot(gs[1, :]), 'H1 Statistic by Trajectory. Note. Column = corridor p95 of each arm’s per-leg peak height above the boom-tip, truth channel, σ = 0.25; whisker = its two-sided 95% seed-cluster bootstrap interval; n = legs that entered the corridor. Dotted orange line = 2.00 m H1 bar; solid red line = 3.42 m DHC-2 airframe contact plane; shaded band = the 1.42 m built-in margin. Dark columns = with the 0.25 m offset (S1O, A2O, PIDO); light columns = without it (S1, A2, PID). Lower is better. Each panel holds one DHC-2 trajectory, nine wind stages pooled, 270 legs or same-seed pairs per arm. No trajectory panel is graded; the hypotheses are graded on both trajectories pooled. Amplifying; decides no hypothesis.', S.TEXT_W_LANDSCAPE)
    S.save_exact(fig, "H1_BY_TRAJECTORY", S.TEXT_W_LANDSCAPE)
    S.FONT_ROW, M1.WHITE_FONT = keep, keep_w
    # H2 and H3: straight on top, racetrack below
    for mod, kind, stem, ttl, extra, h in ((M2, "H2", "H2_BY_TRAJECTORY", "Same-Seed Difference in Docking-Envelope Captures Against the PID-Baseline by DHC-2 Trajectory, Light Turbulence (σ = 0.25)", "σ = 0.25", 3.6),
                                           (M3, "H3", "H3_BY_TRAJECTORY", "Same-Seed Decrease in Docking-Envelope Captures From Light (σ = 0.25) to Moderate (σ = 0.35) Turbulence by DHC-2 Trajectory", "σ = 0.25 to σ = 0.35", 3.6)):
        cells = paired_cells(mod, kind); w = {k: NG for k in mod.ROWS}
        fig = plt.figure(figsize=(S.TEXT_W_PORTRAIT, 2 * h + 0.85)); gs = fig.add_gridspec(3, 1, height_ratios=[h, h, 0.85]); fig.set_layout_engine("constrained")
        for i, t in enumerate(TRAJ):
            ax = fig.add_subplot(gs[i, 0]); mod.draw_panel(fig, ax, cells, w, t); title(ax, t, extra)
        S.crop_note(fig, fig.add_subplot(gs[2, 0]), {"H2": 'H2 Statistic by Trajectory. Note. Capture = truth and perception both inside M3b for 3.0 s (the “Decision Rules” section). Marker = each arm’s same-seed difference in docking-envelope captures from the PID-baseline, σ = 0.25; whisker = its two-sided 95% seed-cluster bootstrap interval. The rate above each marker is the arm’s capture rate on that trajectory; the value below it is the difference; the interval ends are under the whisker caps. Dotted green line = 0 pp, the PID-baseline. Hollow marker = with the 0.25 m offset; filled = without. The 35% reference floor is defined on both trajectories pooled and is not drawn. Each panel holds one DHC-2 trajectory, nine wind stages pooled, 270 legs or same-seed pairs per arm. No trajectory panel is graded; the hypotheses are graded on both trajectories pooled. Amplifying; decides no hypothesis.', "H3": 'H3 Statistic by Trajectory. Note. Capture = truth and perception both inside M3b for 3.0 s (the “Decision Rules” section). Marker = each arm’s same-seed decrease in docking-envelope captures from light (σ = 0.25) to moderate (σ = 0.35) turbulence; whisker = its two-sided 95% seed-cluster bootstrap interval. The rates above each marker are the arm’s capture rates on that trajectory at the two levels; the value below it is the decrease. Dotted green line = 35 pp. Hollow marker = with the 0.25 m offset; filled = without. Each panel holds one DHC-2 trajectory, nine wind stages pooled, 270 legs or same-seed pairs per arm. No trajectory panel is graded; the hypotheses are graded on both trajectories pooled. Amplifying; decides no hypothesis.'}[kind], S.TEXT_W_PORTRAIT)
        S.save_exact(fig, stem, S.TEXT_W_PORTRAIT)
    d = pd.DataFrame(rows_out); stem = os.path.join(S.FIGDIR, "HYPOTHESES_BY_TRAJECTORY"); d.to_csv(stem + ".csv", index=False)
    open(stem + ".md", "w").write("# Hypothesis statistics by DHC-2 trajectory (figures H1_/H2_/H3_BY_TRAJECTORY) - POST-PLAN PRESENTATION; decides nothing; no row is graded\n\n" + S.md(d) + "\n")
    print("wrote", stem + ".csv|.md"); return 0


if __name__ == "__main__":
    sys.exit(main())
