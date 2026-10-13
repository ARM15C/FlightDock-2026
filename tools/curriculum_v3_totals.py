#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: Curriculum-v3.1 RC5 GENERATOR-OF-RECORD — emits the signed
#   stage table and ALL FOUR A9 totals (nominal learner / hard-max learner /
#   evaluation simulator / wall-clock) computed from the table alone.
# PUBLIC API: stage_table() -> list[dict]; retention_strata_count(name);
#   totals() -> dict; render_markdown(t) -> str. CLI: [--json PATH]
#   [--markdown PATH] [--check].
# SIDE-EFFECTS: writes --json/--markdown paths only; no repo/source reads.
# CAPS: file soft 200 / hard 240 LoC; function cap 60.
# COVERAGE: tests/unit/test_curriculum_v3_totals.py (pins 928k / 1,796k /
#   67 stratum-screens / 946 nominal eval episodes / anchor set / ladder).
# =============================================================================
import argparse
import json

H_BAR = 2400                       # full-length episode transitions (A9)
# Per-stage sigma is a COLUMN since the SIGNED diagonal (R1, 2026-08-05);
# taxonomy of record: light=0.25 / moderate=0.35
CONSTANTS = {"wind_from_deg": 180.0,
             "L_m": (300.0, 300.0, 300.0), "cep_m": 0.25,
             "latency": "#379 constant [PIN at P2a]",
             "trajectory_split": "50/50 straight/racetrack"}
# H2's decisive cell pooled {6,7,8}
ANCHORS = ("S0", "W8", "C1")
GATE_N = {"mid12": 12, "anchor30": 30}
# H3 owns the generalization claim.
CERT_N = 50


def stage_table():
    """The DECIDED ladder = the SIGNED DIAGONAL (R1, E3=d, 2026-08-05):
    RC1 TILTED + RC2 C1-160k + T-190 WS5 append + A5 D1/D2 inserts."""
    rows = [dict(name="S0", wind_ms=0.0, turb=False, sigma=0.10,
                 nominal=60_000,
                 cap=120_000, cap_note="S0 cap 120k SIGNED (R23 2026-07-18)",
                 gate="anchor30")]
    rows.append(dict(name="W0", wind_ms=0.0, turb=True, sigma=0.10,
                     nominal=36_000, cap=96_000, gate="mid12"))
    for k in range(1, 13):                            # W1..W12 @ sigma 0.10
        rows.append(dict(name=f"W{k}", wind_ms=round(0.5 * k, 2), turb=True,
                         sigma=0.10,
                         nominal=36_000 if k <= 8 else 48_000,
                         cap=96_000,
                         gate="anchor30" if f"W{k}" in ANCHORS else "mid12"))
    rows.append(dict(name="D1a", wind_ms=6.0, turb=True, sigma=0.15,
                     nominal=48_000, cap=96_000, gate="mid12"))
    rows.append(dict(name="D1", wind_ms=6.0, turb=True, sigma=0.20,
                     nominal=48_000, cap=96_000, gate="mid12"))
    for k in range(13, 17):                           # W13..W16 @ sigma 0.20
        rows.append(dict(name=f"W{k}", wind_ms=round(0.5 * k, 2), turb=True,
                         sigma=0.20, nominal=48_000, cap=96_000,
                         gate="mid12"))
    rows.append(dict(name="D2a", wind_ms=8.0, turb=True, sigma=0.25,
                     nominal=48_000, cap=96_000, gate="mid12"))
    rows.append(dict(name="D2b", wind_ms=8.0, turb=True, sigma=0.30,
                     nominal=48_000, cap=96_000, gate="mid12"))
    rows.append(dict(name="C1",
                     wind_ms="stratified 32-cell (W1..W16 x traj; "
                             "D1a/D1/D2a/D2b excluded — duplicate winds)",
                     turb=True, sigma=0.30,           # FIXED (election E1=a)
                     nominal=160_000, cap=320_000, gate="anchor30"))
    for nm, w in (("D1b", 6.0), ("R13", 6.5), ("R14", 7.0), ("R15", 7.5),
                  ("L4", 4.0), ("L2", 2.0), ("L0", 0.0)):
        rows.append(dict(name=nm, wind_ms=w, turb=True, sigma=0.25,
                         nominal=48_000, cap=72_000, gate="mid12"))
    for nm, w, sg in (("P1a", 6.0, 0.15), ("P1", 6.0, 0.20),
                      ("L3", 3.0, 0.25), ("P2a", 8.0, 0.25)):
        rows.append(dict(name=nm, wind_ms=w, turb=True, sigma=sg,
                         nominal=48_000, cap=72_000, gate="mid12"))
    for nm, w, sg in (("A2A0", 0.0, 0.0), ("A2A8", 8.0, 0.0),
                      ("A2A10", 8.0, 0.10), ("A2A15", 8.0, 0.15),
                      ("A2A20", 8.0, 0.20), ("A2A25", 8.0, 0.25)):
        rows.append(dict(name=nm, wind_ms=w, turb=sg > 0.0, sigma=sg,
                         nominal=48_000, cap=72_000, gate="mid12"))
    for r in rows:
        r.update(CONSTANTS)
        r["sigma_ms"] = (r["sigma"],) * 3
    return rows


def ladder_names():
    return [r["name"] for r in stage_table()]


def retention_strata_count(name):
    """A7/RC3 panel size at stage `name`'s gate: {S0, W0, previous rung, one
    older mastered W-rung rotation} + ALL prior anchors at anchor gates."""
    names = ladder_names()
    i = names.index(name)
    mastered = names[:i]
    panel = [s for s in ("S0", "W0") if s in mastered]
    if mastered and mastered[-1] not in panel:
        panel.append(mastered[-1])                       # previous rung
    rot = [s for s in mastered if s.startswith("W") and s != "W0"
           and s not in panel]
    if rot:
        panel.append(rot[0])                             # one rotation slot
    if name in ANCHORS:
        panel += [a for a in ANCHORS[:ANCHORS.index(name)] if a not in panel]
    return len(panel)


def totals():
    """The four A9 totals, generator-computed (RC5)."""
    rows = stage_table()
    nominal = sum(r["nominal"] for r in rows)
    cap_sum = sum(r["nominal"] if r["cap"] is None else r["cap"] for r in rows)
    pool = 1_796_000                     # R23: previously-signed number, kept
    gate_eps = sum(GATE_N[r["gate"]] for r in rows) + CERT_N
    strata = {r["name"]: retention_strata_count(r["name"]) for r in rows}
    ret_screens = sum(strata.values())
    ret_eps = 6 * ret_screens
    pid_eps = 30 * len(ANCHORS) + CERT_N                 # A10 paired arm
    eval_eps = gate_eps + ret_eps + pid_eps
    return {
        "package": "RC1 TILTED (operator-decided 2026-07-17) + RC2 C1=160k",
        "h_bar_qualifier": ("ALL episode-derived figures at full-length "
                            f"H-bar={H_BAR}; P2b measured H-bar recalibrates"),
        "learner_nominal_transitions": nominal,
        "learner_pool_inclusive_transitions": pool,
        "conditional_reserve_transitions": pool - nominal,
        "learner_stage_cap_sum_transitions": cap_sum,
        "pool_convention": ("R23 SIGNED: pool is BINDING + INCLUSIVE (all "
                            "conditional training — inserts, re-climbs — "
                            "debits it; monotonic counter, never restored). "
                            "PROTECTED RESERVE: a conditional debit is "
                            "REFUSED unless remaining pool >= sum of nominal "
                            "allocations of all uncompleted canonical stages "
                            "(else HALT class BUDGET_TRUNCATED — a budget "
                            "outcome, not a competence verdict). Per-stage "
                            "cap-sum is diagnostic only; the pool binds."),
        "q4_insert_unit_costs": {"low_band_W0_W8": 36_000,
                                 "high_band_W9_W16": 48_000,
                                 "note": ("R23 scope: W1-W16 ORIGINALS only, "
                                          "<=1 insert each, non-recursive; "
                                          "S0/W0/C1 exhaust => demote-or-HALT; "
                                          "insert training draws from the "
                                          "inclusive pool (no cap growth); "
                                          "unique signed stage ID; joins "
                                          "retention rotation after promotion")},
        "eval_episode_ledger": {
            "gate_screens_anchors_cert": gate_eps,
            "retention_screens": ret_eps,
            "retention_stratum_screen_count": ret_screens,
            "retention_strata_by_gate": strata,
            "pid_paired_arm": pid_eps,
            "nominal_eval_episodes": eval_eps},
        "eval_sim_transitions_nominal": eval_eps * H_BAR,
        "eval_contingency_unit_costs": {
            "retention_confirmation_per_stratum_eps": 12,
            "zero_shot_probe_eps": 6,
            "q4_insert_regate_eps": GATE_N["mid12"],
            "note": "conditional; never summed into nominal (RC3/SectionC)"},
        "wall_clock": {
            "formula": "hours = transitions / measured_steps_per_hour",
            "measured_steps_per_hour": None,
            "status": "P2b-measured; placeholders below are [DERIVED indicative]",
            "indicative": {"n4_1x_18k_per_h_learner_nominal_h": round(
                               nominal / 18_000, 1),
                           "design_base_n16_2x_collection_h": "6.5-8 (tilted)"}},
    }


def render_markdown(t):
    rows = stage_table()
    out = ["# Curriculum v3.1 stage table + A9 totals (RC5 generator-of-record)",
           f"_{t['package']}; {t['h_bar_qualifier']}_", "",
           "| stage | wind m/s | turb σ/L | nominal | cap | gate |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        turb = "0.75/300" if r["turb"] else "off"
        cap = r["cap"] if r["cap"] is not None else "[P2b]"
        out.append(f"| {r['name']} | {r['wind_ms']} | {turb} | "
                   f"{r['nominal']:,} | {cap if isinstance(cap, str) else format(cap, ',')} | {r['gate']} |")
    out += ["", f"- learner nominal: **{t['learner_nominal_transitions']:,}**",
            f"- learner INCLUSIVE POOL (R23 SIGNED, binding): "
            f"**{t['learner_pool_inclusive_transitions']:,}** (reserve "
            f"{t['conditional_reserve_transitions']:,}; stage-cap sum "
            f"{t['learner_stage_cap_sum_transitions']:,} diagnostic-only)",
            f"- eval nominal: **{t['eval_episode_ledger']['nominal_eval_episodes']}"
            f" episodes = {t['eval_sim_transitions_nominal']:,} transitions** "
            f"(retention stratum-screens: "
            f"{t['eval_episode_ledger']['retention_stratum_screen_count']})",
            f"- wall-clock: {t['wall_clock']['status']}"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Curriculum v3.1 RC5 totals generator")
    ap.add_argument("--json", default=None)
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    t = totals()
    if args.check:
        ok = (t["learner_nominal_transitions"] == 1_936_000     # +11 x 48k, A2A +6 x 48k
              and t["learner_pool_inclusive_transitions"] == 1_796_000  # unchanged BASE
              and t["learner_stage_cap_sum_transitions"] == 3_680_000  # +11 x 72k, A2A +6 x 72k
              and t["conditional_reserve_transitions"] == -140_000  # stale-base diagnostic
              and t["eval_episode_ledger"]["retention_stratum_screen_count"] == 151)
        print(f"check: {'OK' if ok else 'MISMATCH'} {t['learner_nominal_transitions']}")
        raise SystemExit(0 if ok else 1)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"stage_table": stage_table(), "totals": t}, f, indent=1)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(render_markdown(t))
    if not (args.json or args.markdown):
        print(json.dumps(t, indent=1))


if __name__ == "__main__":
    main()
