#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: answer "did this terminal's CONDITION fire on this leg", for
#   a terminal that TERMDEMOTE-767 may have demoted. This is the ONE WRITER of
#   that disjunction for the three consumers of leg_terminal_reader
#   (gate_safety_events, curriculum_eval_spawner, gate_leg_report).
#
#   WHY IT EXISTS. Before demotion, "SAFETY_COLLISION in terminal_types" was a
#   complete test: the condition and the terminal were the same event. After
#   demotion the condition fires and the leg FLIES ON, so that test reads
#   structurally False on exactly the legs that breached -- silently flipping
#   hard_fail_v1/v2/v3/v4, the panel rollup, the ARM-B SAP scorer and M-8/9/10.
#   The predicate is UNCHANGED; only its evidence moved. Putting the OR here
#   instead of at three call sites means a future terminal joins the demote set
#   without touching any consumer.
#
#   WHY NOT INSIDE leg_terminal_reader. That module answers "what terminal did
#   the env DECLARE" and its cap is set to its own size so any growth alarms --
#   a deliberate design. This answers a different question and carries its own
#   cap.
#
# CHANNELS: results_eid*.json episodes[] -> `demoted_<short>_ticks` (PRIMARY),
#   metrics_*.csv same column (FALLBACK). Mirrors leg_terminal_reader's
#   preference order exactly.
# FAIL DIRECTION: None, never False, when NO channel is readable -- "unknown"
#   must not be readable as "no event". A readable leg with the columns ABSENT
#   is a PRE-TERMDEMOTE leg: that is not unknown, its terminal channel is the
#   whole truth, so it reads 0. Absent and unreadable are different answers.
# PUBLIC API: demoted_stat/demoted_ticks/demoted_events -> int|None;
#   condition_fired(leg_dir, terminal) -> bool|None; main(argv). CLI: --leg-dir.
# SIDE-EFFECTS: none (pure reader; CLI prints JSON).
# CAPS: file soft 90 / hard 120 code-lines; function cap 35.
# COVERAGE: tests/unit/test_leg_demoted_reader.py, both directions.
# =============================================================================
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leg_terminal_reader as _ltr  # noqa: E402

# terminal -> the column short name the env emits (env._DEMOTE_FIELDS)
SHORT = {"SAFETY_COLLISION": "collision", "SAFETY_AIRSPEED": "airspeed",
         "SAFETY_CONTACT": "contact", "SAFETY_NOPASS": "nopass"}


STATS = ("ticks", "intervals")


def _col(terminal, stat="ticks"):
    return "demoted_%s_%s" % (SHORT[terminal], stat)


def _from_results(leg_dir, col):
    """Max ticks over episodes, or None if no file yields the column."""
    vals = []
    for p in sorted(glob.glob(os.path.join(leg_dir, "results_eid*.json"))):
        try:
            eps = json.load(open(p)).get("episodes") or []
        except (OSError, ValueError):
            continue
        vals += [e[col] for e in eps
                 if isinstance(e, dict) and isinstance(e.get(col), (int, float))]
    return max(vals) if vals else None


def _from_metrics(leg_dir, col):
    vals = []
    for p in sorted(glob.glob(os.path.join(leg_dir, "metrics_*.csv"))):
        try:
            for r in csv.DictReader(open(p)):
                if (r.get(col) or "").strip():
                    vals.append(float(r[col]))
        except (OSError, KeyError, ValueError):
            continue
    return max(vals) if vals else None


def demoted_stat(leg_dir, terminal, stat="ticks"):
    """`ticks` the demoted condition held, or the count of separate
    `intervals` it held over. 0 when the columns are ABSENT (a pre-TERMDEMOTE
    leg -- its terminal channel is the whole truth). None ONLY when no channel
    is readable at all, which is a different answer and must stay one."""
    assert stat in STATS, stat
    if _ltr.terminal_types(leg_dir) is None:
        return None
    col = _col(terminal, stat)
    got = _from_results(leg_dir, col)
    if got is None:
        got = _from_metrics(leg_dir, col)
    return int(got) if got is not None else 0


def demoted_ticks(leg_dir, terminal):
    return demoted_stat(leg_dir, terminal, "ticks")


def demoted_events(leg_dir, terminal):
    """How many SEPARATE times the condition fired -- the honest event COUNT
    for a demoted leg, where one terminal used to mean exactly one event."""
    return demoted_stat(leg_dir, terminal, "intervals")


def condition_fired(leg_dir, terminal):
    """THE predicate: did this terminal's condition fire, whether it ended the
    episode or was demoted to an event? None when the leg is unreadable."""
    terms = _ltr.terminal_types(leg_dir)
    if terms is None:
        return None
    return bool(terminal in terms or (demoted_ticks(leg_dir, terminal) or 0) > 0)


# The terminals a gate SCORES, as (limb_name, terminal_string). This tuple is
# the ONLY edit a new scored terminal needs: gate_safety_events.score_leg
# consumes terminal_limbs() and therefore does NOT grow when a terminal is
# added. That matters because score_leg sits at 58 of a 60-line function cap
# and its file at 310 of a 310-line hard cap, so growth there is a breach, not
# a cost. Lives HERE, beside condition_fired, because this module is already
# the ONE WRITER of the terminal-OR-demoted-event disjunction.
# NOPASSTERM-770: neither new terminal is demotable, so condition_fired
# reduces to the terminal channel for them; the limb pair still appears so a
# scorer can key on it without growing (hard_fail_v2 is NOT changed here).
SCORED_TERMINALS = (("collision", "SAFETY_COLLISION"),
                    ("contact", "SAFETY_CONTACT"),
                    ("nopass", "SAFETY_NOPASS"))


def terminal_limbs(leg_dir, terms):
    """Per scored terminal: the CONDITION and the TERMINAL-ONLY channel.

    The predicate is unchanged by demotion -- only its evidence moved. Before
    TERMDEMOTE-767 the condition and the terminal were the same event, so a
    membership test was complete; after demotion the condition fires and the
    leg flies on, so the bare test reads False on exactly the legs that
    breached. `<name>` is the CONDITION and is what every hard_fail criterion
    must key on. `<name>_terminal` stays TERMINAL-ONLY so the two never
    silently merge and a re-score can always tell a demoted leg from a
    terminated one.
    """
    out = {}
    for name, tname in SCORED_TERMINALS:
        out[name] = bool(condition_fired(leg_dir, tname))
        out["%s_terminal" % name] = tname in terms
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg-dir", required=True)
    a = ap.parse_args(argv)
    print(json.dumps({t: {"ticks": demoted_ticks(a.leg_dir, t),
                          "events": demoted_events(a.leg_dir, t),
                          "condition_fired": condition_fired(a.leg_dir, t)}
                      for t in sorted(SHORT)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
