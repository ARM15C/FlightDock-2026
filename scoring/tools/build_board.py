#!/usr/bin/env python3
# =============================================================================

# RESPONSIBILITY: build the ONE per-leg board of record, BOARD_W10.csv (schema = the 09-03 CANDBOARD_W10.csv
#   columns, so every existing table/figure script can be re-pointed by path, PLUS the columns the 09-15 plans add),
#   from THIS run's readouts, for all twelve corpora.
#   NO BANKED COMPARISON. The layer-1 limb of
#   board_of_record.py -- which required this run to reproduce the banked readouts cell by cell -- is REMOVED, not
#   skipped: five of the six banked readouts were computed on stripped corpus copies that are now quarantined, and
#   PIDO has no banked readout at all. Every number here is computed from this run's own readouts.
#   The FOUR-CONDITION success flag of the H2 plan (dwell >= 3.0 s on both channels; truth peak <= 2.00 m inside the
#   aerial-docking terminal approach corridor; no overtake past the DHC-2 CG in close formation; no airframe contact)
#   is computed with pandas from the per-leg records and printed BESIDE the exam's own flag with the count of legs
#   that differ, per stage. `passed` in the board IS the flag of record; `passed_exam` is the exam's.
#   Also per stage and corpus: legs that never reached the corridor (no corridor peak; failures in every rate, out
#   of the p95 pool only), overtake legs, no-pass and contact terminals, guard fires, realized sigma and
#   out-of-band legs (kept), held altitude. Turbulence: 0.25 / 0.35 only (T5). A peak over the bar is a
#   safety-EXCLUSION-ZONE violation (T1). "stage" never "rung" in every output.
# PUBLIC API: load(run_dir, tag) -> DataFrame; flag_of_record(df) -> Series; diag(df) -> DataFrame;
#   build(run_dir) -> int; main(argv=None) -> int. CLI: --run-dir DIR [--only TAG,...] [--selftest]
# SIDE-EFFECTS: reads this run's readouts READ-ONLY; writes BOARD_W10.csv, BOARD_W3.csv, DIAG_PER_STAGE.csv and
#   FLAG_COMPARISON.csv under <run-dir>/board/. Spawns nothing. No network, no S3.
# SIZE: soft 200 / hard 240 code-LoC; function cap 45.
# COVERAGE: --selftest: a synthetic corpus where one leg dwells but breaches the bar OUTSIDE the corridor flips to a
#   success of record; a leg breaching INSIDE stays a failure; an overtake leg stays a failure; a no-corridor leg is
#   a failure with a null peak; a corpus that does not hold 600 legs REFUSES.
# =============================================================================
import argparse
import ast
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rescore_run as R  # noqa: E402
import p95_boot as PB  # noqa: E402

BAR, CONTACT, DWELL = 2.00, 3.42, 3.0
SW = ["W12", "W14", "W16"]        # 6, 7 and 8 m/s mean wind


def load(run_dir, tag):
    ro = os.path.join(run_dir, "readouts")
    d = json.load(open(os.path.join(ro, "%s_READOUT.json" % tag)))
    w10 = json.load(open(os.path.join(ro, "WIN10_%s.json" % tag)))
    w3 = json.load(open(os.path.join(ro, "WIN3_%s.json" % tag)))
    hp = os.path.join(ro, "HELD10_%s.json" % tag)
    h = json.load(open(hp)) if os.path.exists(hp) else {"legs": {}}     # absent only in a dry run; the run of record writes it
    df = pd.DataFrame(d["legs"]).merge(pd.DataFrame(w10["legs"])[["leg", "win", "whole"]], on="leg", how="left")
    df = df.merge(pd.DataFrame(w3["legs"])[["leg", "win"]].rename(columns={"win": "win3"}), on="leg", how="left")
    held = pd.DataFrame([{"leg": k, "held_median_above_m": v.get("median_above_m")} for k, v in h["legs"].items()],
                        columns=["leg", "held_median_above_m"])
    df = df.merge(held, on="leg", how="left")
    df["key"] = df.wind_label + "|" + df.seed.astype(str) + "|" + df.trajectory
    df["wind_rung"] = df.wind_label != "S0"      # column NAME kept for the 09-03 scripts; it means "wind stage"
    df["sw"] = df.wind_label.isin(SW)
    df["corpus"] = tag
    if len(df) != 600:
        raise SystemExit("REFUSE rc 1: %s holds %d legs, not 600" % (tag, len(df)))
    return df


def flag_of_record(df):
    """The H2 plan's four conditions, read from the per-leg records (never from the exam's own verdict)."""
    dwell = pd.to_numeric(df.intersection_v3_s, errors="coerce").fillna(0) >= DWELL
    corridor_ok = df.win.notna() & (df.win <= BAR)          # a leg that never entered has no peak: not a success
    no_overtake = ~df.real_overtake.astype(bool)
    terms = df.terminals.apply(lambda t: t if isinstance(t, dict) else (ast.literal_eval(t) if isinstance(t, str) and t else {}))
    no_contact = ~terms.apply(lambda t: "SAFETY_CONTACT" in t)
    return dwell & corridor_ok & no_overtake & no_contact


def diag(df):
    """Per stage: everything the plans list as reported diagnostics, for one corpus."""
    terms = df.terminals.apply(lambda t: t if isinstance(t, dict) else (ast.literal_eval(t) if isinstance(t, str) and t else {}))
    guards = df.guards.apply(lambda g: g if isinstance(g, dict) else (ast.literal_eval(g) if isinstance(g, str) and g else {}))
    gate = df.sigma_gate.apply(lambda g: g if isinstance(g, dict) else (ast.literal_eval(g) if isinstance(g, str) and g else {}))
    x = pd.DataFrame({"corpus": df.corpus, "stage": df.wind_label, "cell": df.cell,
                      "success_record": df.passed, "success_exam": df.passed_exam, "entered_corridor": df.win.notna(),
                      "over_bar_corridor": df.win > BAR, "contact_corridor": df.win > CONTACT, "contact_whole": df.whole > CONTACT,
                      "overtake_leg": df.real_overtake.astype(bool),
                      "nopass_terminal": terms.apply(lambda t: int(t.get("SAFETY_NOPASS", 0)) > 0),
                      "alt_guard_fires": guards.apply(lambda g: int((g.get("by_type") or {}).get("guard_altitude_ceiling", 0))),
                      "overtake_guard_fires": guards.apply(lambda g: int((g.get("by_type") or {}).get("guard_hard_overtake", 0))),
                      "oob_sigma": gate.apply(lambda g: g.get("state") not in (None, "MEASURED_PASS", "CALM_CONFIRMED")),
                      "win": df.win, "seed": df.seed})
    g = x.groupby(["corpus", "stage"], sort=False)
    out = g.agg(n=("success_record", "size"), k_record=("success_record", "sum"), k_exam=("success_exam", "sum"),
                entered_corridor=("entered_corridor", "sum"), never_reached_corridor=("entered_corridor", lambda s: int((~s).sum())),
                over_bar_corridor=("over_bar_corridor", "sum"), contact_corridor=("contact_corridor", "sum"),
                contact_whole=("contact_whole", "sum"), overtake_legs=("overtake_leg", "sum"),
                nopass_terminals=("nopass_terminal", "sum"), alt_guard_fires=("alt_guard_fires", "sum"),
                overtake_guard_fires=("overtake_guard_fires", "sum"), oob_sigma_legs=("oob_sigma", "sum"),
                p95_corridor=("win", PB.p95)).reset_index()
    out["p95_corridor_ci95"] = [PB.fmt(*PB.ci(s, "win")) for _, s in g]      # per-stage p95: same seed-cluster bootstrap
    out["flag_changed_legs"] = (g.apply(lambda s: int((s.success_record != s.success_exam).sum()))).values
    return out


def build(run_dir, tags=None):
    frames = []
    for tag in (tags or list(R.CORPORA)):
        df = load(run_dir, tag)
        df["passed_exam"] = df.passed.astype(bool)
        df["passed"] = flag_of_record(df)
        frames.append(df)
    out = os.path.join(run_dir, "board")
    os.makedirs(out, exist_ok=True)
    board = pd.concat(frames, ignore_index=True)
    board.to_csv(os.path.join(out, "BOARD_W10.csv"), index=False)
    b3 = board.copy()
    b3["win"] = b3.win3
    b3.to_csv(os.path.join(out, "BOARD_W3.csv"), index=False)
    d = pd.concat([diag(f) for f in frames], ignore_index=True)
    d.to_csv(os.path.join(out, "DIAG_PER_STAGE.csv"), index=False)
    cmp_ = d.groupby("corpus")[["n", "k_record", "k_exam", "flag_changed_legs", "never_reached_corridor",
                                "over_bar_corridor", "overtake_legs", "oob_sigma_legs"]].sum()
    cmp_.to_csv(os.path.join(out, "FLAG_COMPARISON.csv"))
    print(cmp_.to_string())
    return 0


def selftest():
    rows = []
    for i, (dw, win, whole, ot, term, exam) in enumerate([(5.0, 1.0, 2.5, False, {}, False), (5.0, 2.2, 2.2, False, {}, False),
                                                        (5.0, 1.0, 1.0, True, {}, False), (5.0, np.nan, 1.0, False, {}, False),
                                                        (1.0, 1.0, 1.0, False, {}, False), (5.0, 1.0, 1.0, False, {}, True)]):
        rows.append({"leg": "W16_%d_straight" % i, "intersection_v3_s": dw, "win": win, "whole": whole, "real_overtake": ot,
                     "terminals": term, "passed": exam})
    df = pd.DataFrame(rows)
    f = flag_of_record(df).tolist()
    ok = f == [True, False, False, False, False, True]
    ok = ok and len(R.CORPORA) == 12 and "PIDO25" in R.CORPORA
    print("SELFTEST", "PASS" if ok else "FAIL", f)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--only")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.run_dir:
        ap.error("--run-dir required")
    return build(a.run_dir, a.only.split(",") if a.only else None)


if __name__ == "__main__":
    sys.exit(main())
