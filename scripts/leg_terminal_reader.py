#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: resolve the ENVIRONMENT'S OWN episode terminal type(s) for a
#   banked eval leg dir. This is the ONE WRITER
#   of "what terminal did the env declare" for THREE consumers: the v2 gate
#   criterion (gate_safety_events), the legacy summary writer
#   (curriculum_eval_spawner) and the panel abort classifier (gate_leg_report).
#   It exists because all three read TRANSITION_SNAPSHOT.log for the token
#   SAFETY_COLLISION — a token that file CANNOT contain (its vocabulary is FSM
#   states; SAFETY_COLLISION is the env's episode TERMINATOR, written to
#   results_eid*.json episodes[].terminal_type and metrics_*.csv terminal_type).
#   A guard that cannot fire is not a guard.
# CHANNELS, in preference order: results_eid*.json -> episodes[].terminal_type
#   (structured, PRIMARY); metrics_*.csv -> terminal_type column (FALLBACK).
#   When both are readable and disagree, the JSON wins AND the disagreement is
#   exposed (channels_disagree) — never resolved silently.
# FAIL DIRECTION: no readable channel => None, NEVER []. None is meaningful:
#   "unknown" must not be readable as "no collision"; callers fail closed.
# PUBLIC API: terminal_types(leg_dir, detail=False) -> list|None (detail=True
#   -> dict with source + channels_disagree); has_terminal(leg_dir, token) ->
#   bool|None; selfcheck() -> int; main(argv). CLI: --leg-dir DIR | --selfcheck.
# SIDE-EFFECTS: none (pure reader; CLI prints JSON). Selfcheck tempdir removed
#   in finally.
# CAPS: file soft 120 / hard 140 code lines; function cap 40.
#   (Plan r2 declared soft 90 / hard 120 from my estimate; as-built measured
#   115 — the third consecutive underestimate of instrument+battery cost, on
#   the record. Set at FIRST COMMIT, before any attached to
#   these numbers; soft 120 = current size, so ANY growth trips the alarm.)
# COVERAGE: tests/unit/test_leg_terminal_reader.py + --selfcheck (both
#   directions) + the consumer batteries of S-10/S-12/S-13.
# =============================================================================
import argparse
import csv
import glob
import json
import os
import sys


def _from_results(leg_dir):
    """episodes[].terminal_type from results_eid*.json, in file/episode order,
    or None if no file yields one. The KEY MATTERS: only `terminal_type` is
    accepted — a renamed or misspelled key reads as absent, never guessed."""
    vals = []
    for p in sorted(glob.glob(os.path.join(leg_dir, "results_eid*.json"))):
        try:
            eps = json.load(open(p)).get("episodes") or []
        except (OSError, ValueError):
            continue
        vals += [e["terminal_type"] for e in eps
                 if isinstance(e, dict)
                 and isinstance(e.get("terminal_type"), str)]
    return vals or None


def _from_metrics(leg_dir):
    """terminal_type column from metrics_*.csv (fallback channel)."""
    vals = []
    for p in sorted(glob.glob(os.path.join(leg_dir, "metrics_*.csv"))):
        try:
            vals += [r["terminal_type"] for r in csv.DictReader(open(p))
                     if r.get("terminal_type")]
        except (OSError, KeyError):
            continue
    return vals or None


def terminal_types(leg_dir, detail=False):
    """The env's own terminal type(s) for this leg, or None if UNREADABLE."""
    rj, mc = _from_results(leg_dir), _from_metrics(leg_dir)
    types = rj if rj is not None else mc
    if not detail:
        return types
    return {"terminal_types": types,
            "source": ("results_json" if rj is not None
                       else "metrics_csv" if mc is not None else None),
            "channels_disagree": (sorted(rj) != sorted(mc)
                                  if rj is not None and mc is not None
                                  else None)}


def has_terminal(leg_dir, token):
    """True/False when readable; None (NOT False) when no channel is readable
    — the caller decides the fail direction, this module never does."""
    types = terminal_types(leg_dir)
    return None if types is None else token in types


def _mk(t, name, js=None, cs=None, key="terminal_type"):
    """Selfcheck fixture leg (JSON and/or CSV channel, controllable key)."""
    d = os.path.join(t, name)
    os.makedirs(d)
    if js is not None:
        json.dump({"episodes": [{key: v} for v in js]},
                  open(os.path.join(d, "results_eid1.json"), "w"))
    if cs is not None:
        open(os.path.join(d, "metrics_1.csv"), "w").write(
            "episode,terminal_type\n"
            + "".join(f"{i},{v}\n" for i, v in enumerate(cs)))
    return d


def _cases(t):
    """The selfcheck battery, BOTH directions per pin."""
    dis = terminal_types(_mk(t, "c", js=["SAFETY_COLLISION"],
                             cs=["TIMEOUT"]), detail=True)
    return [
        ("i JSON primary channel read, multi-episode order kept",
         terminal_types(_mk(t, "a", js=["TIMEOUT", "SAFETY_COLLISION"]))
         == ["TIMEOUT", "SAFETY_COLLISION"]),
        ("ii CSV fallback when no JSON",
         terminal_types(_mk(t, "b", cs=["SAFETY_RANGE"]))
         == ["SAFETY_RANGE"]),
        ("iii disagreement: JSON wins AND is EXPOSED, never silent",
         dis["terminal_types"] == ["SAFETY_COLLISION"]
         and dis["source"] == "results_json"
         and dis["channels_disagree"] is True),
        ("iv agreement reads disagree=False (the flag can say both)",
         terminal_types(_mk(t, "d", js=["TIMEOUT"], cs=["TIMEOUT"]),
                        detail=True)["channels_disagree"] is False),
        ("v no channel => None NEVER []; has_terminal => None NOT False",
         terminal_types(_mk(t, "e")) is None
         and has_terminal(os.path.join(t, "e"), "SAFETY_COLLISION") is None),
        ("vi wrong JSON key reads as ABSENT (falls to CSV), not guessed",
         terminal_types(_mk(t, "f", js=["SAFETY_COLLISION"],
                            cs=["TIMEOUT"], key="terminal_typo"))
         == ["TIMEOUT"]),
        ("vii has_terminal discriminates BOTH directions",
         has_terminal(_mk(t, "g", js=["SAFETY_COLLISION"]),
                      "SAFETY_COLLISION") is True
         and has_terminal(_mk(t, "h", js=["TIMEOUT"]),
                          "SAFETY_COLLISION") is False),
    ]


def selfcheck():
    import shutil
    import tempfile
    t = tempfile.mkdtemp(prefix="ltr_sc_")
    try:
        checks = _cases(t)
        for n, p in checks:
            print(("PASS " if p else "FAIL ") + n)
        print("selfcheck:", "PASS" if all(p for _, p in checks) else "FAIL")
        return 0 if all(p for _, p in checks) else 1
    finally:
        shutil.rmtree(t, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg-dir")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args(argv)
    if a.selfcheck:
        return selfcheck()
    if not a.leg_dir:
        ap.error("--leg-dir required (or --selfcheck)")
    print(json.dumps(terminal_types(a.leg_dir, detail=True), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
