#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: The EVIDENCE-ADMISSIBILITY layer of gate adjudication —
#   "may this leg be graded at all?", answered BEFORE and independently of
#   "did it pass?". Extracted VERBATIM from curriculum_gate_runner.py by
#   T-353 (STRUCTURAL, option (b)); the runner
#   was 359 code-LoC against soft 320 and caps are never raised silently.
#
#   *** STRUCTURAL: ZERO BEHAVIOR DELTA. *** Every function body, comment,
#   constant and docstring below is a byte-level move, not a rewrite — with
#   ONE disclosed exception: `csv`/`glob` were function-local imports in the
#   runner and are hoisted to module scope here (same modules, same objects,
#   no rule touched). No rule was tightened, loosened, reordered or renamed.
#   Leading underscores are dropped on the four private helpers because they
#   are this module's public surface; the runner re-exports them under their
#   ORIGINAL names, so no caller sees a rename. The runner re-exports
#   every moved name, so `curriculum_gate_runner.split_evidence` and friends
#   still resolve exactly as before and no consumer changed.
#
#   The layer boundary is real, not a convenience cut: these rules decide
#   ADMISSIBILITY from a leg's own artifacts (are the instruments present,
#   is the capture complete, is the frozen chain's view bridged). They take
#   no bar, no gate type, no wilson_lb, and no trajectory balance — those
#   belong to the adjudicator that imports this module. The dependency runs
#   one way: runner -> rules. Never back.
# PUBLIC API: split_evidence(rows), split_short_captures(rows),
#   evidence_problem(excluded), short_problem(excluded),
#   unbridged(leg_dir), naive_m3b_best(leg_dir);
#   constants MIN_CAPTURE_S, SETUP_SCREEN_REASON, NAIVE_GAP_TOL_S,
#   NAIVE_DWELL_S.
# SIDE-EFFECTS: none (pure readers over banked leg artifacts; no writes, no
#   env mutation, no subprocess, no network).
# CAPS: file soft 160 / hard 220 code-LoC; function cap 45.
# COVERAGE: tests/unit/test_curriculum_gate_runner.py exercises every moved
#   name THROUGH the runner's re-exports (unchanged, deliberately not
#   rewritten — a passing untouched test is the equivalence evidence), plus
#   the T-353 byte-level pre/post equivalence harness on 14 real banked legs.
# =============================================================================
import csv
import glob
import os

# T-270/T-271 capture-completeness screen. Nominal paired window = 600 s;
# honest observed minimum 544.7 s (W12 43159, ordinary episode-end variation);
# the wrapper-exit kills produced 38-39 s and 334 s captures that were graded
# as valid failures. 500 clears the honest minimum with margin while catching
# every kill shape observed. One fact, one name — pinned by unit test.
# S-4 CORRECTION: NECESSARY but NOT SUFFICIENT. A
# short span has two causes — the RECORDING was cut (evidence loss) or the
# FLIGHT ended early (a real outcome) — and span alone cannot separate them.
# D1 attempt_001 excluded its two WORST safety failures (a 20.43 m real
# overtake; a SAFETY_COLLISION terminal) on a symptom unrelated to their
# failure, turning a gradable FAIL into an INVALID panel and hiding 2 of the
# 3 H1 ceiling busts. The screen now ALSO requires capture_complete==False
# (gate_safety_events, fail-closed when the terminal time is unreadable).
# Adding a conjunct can only SHRINK the excluded set: new is a SUBSET of old,
# so any verdict that excluded nothing under the old rule is unchanged.
MIN_CAPTURE_S = 500.0
# E-4 reserved evidence class — the setup screen HOOK. The generic evidence
# machinery below handles any reason carrying this name; NO code emits it yet:
# the firing criterion must be pre-registered from the 43435 RCA output first.
SETUP_SCREEN_REASON = "INDETERMINATE_SETUP"
NAIVE_GAP_TOL_S = 0.25          # dwell convention of record (gap tolerance)
NAIVE_DWELL_S = 3.0


def unbridged(leg_dir):
    """E-4 refuse-to-grade-unbridged (the T-273 defect, mechanized): the leg
    HAS spawner-side env CSVs but the frozen chain's paired/env0 view is
    absent — grading it would score an evidence gap as a flight failure."""
    has_env = bool(glob.glob(os.path.join(leg_dir, "env*",
                                          "paired_channels_ep*.csv")))
    bridge = os.path.join(leg_dir, "paired", "env0",
                          "paired_channels_ep0001.csv")
    return has_env and not os.path.exists(bridge)


def naive_m3b_best(leg_dir):
    """E-4 second reader: best gap-tolerant contiguous both-mask (raw_m3b AND
    ctrl_m3b) dwell from the leg's own env CSVs. Independent of the frozen
    chain (flag columns vs signed-L̂ masks). Returns None when the flag
    columns are absent. Used ONLY to annotate CONTRADICTION on a null frozen
    read — it can stop a false FAIL, never mint a PASS."""
    ticks = []
    for p in sorted(glob.glob(os.path.join(leg_dir, "env*",
                                           "paired_channels_ep*.csv"))):
        with open(p, newline="") as fh:
            rd = csv.DictReader(fh)
            if not rd.fieldnames or "raw_m3b" not in rd.fieldnames \
                    or "ctrl_m3b" not in rd.fieldnames:
                continue
            for r in rd:
                try:
                    t = float(r["sim_time_s"])
                except (TypeError, ValueError, KeyError):
                    continue
                on = (r.get("raw_m3b") in ("1", "True", "true")
                      and r.get("ctrl_m3b") in ("1", "True", "true"))
                ticks.append((t, on))
    if not ticks:
        return None
    ticks.sort()
    best = cur0 = prev = None
    for t, on in ticks:
        if on and (cur0 is None or t - prev > NAIVE_GAP_TOL_S):
            cur0 = t
        if on:
            prev_span = t - cur0
            best = prev_span if best is None or prev_span > best else best
            prev = t
        else:
            cur0 = None
    return round(best, 3) if best is not None else 0.0


def split_evidence(rows):
    """E-4: legs whose instruments could not measure the flight are
    INDETERMINATE — excluded from k/n, named, and the panel fails CLOSED via
    invalid_protocol until they are replaced or repaired. Both directions of
    fail-closed preserved: an evidence leg can never PASS and is never a
    counted performance miss."""
    gradable = [r for r in rows if not r.get("evidence_problems")]
    excluded = [r for r in rows if r.get("evidence_problems")]
    return gradable, excluded


def evidence_problem(excluded):
    return "INDETERMINATE_EVIDENCE:" + ",".join(
        "%s(%s)" % (r["leg"], "|".join(r["evidence_problems"]))
        for r in excluded)


def split_short_captures(rows):
    """T-270 §7.1 ruling, CORRECTED by S-4: an external kill can DESTROY
    evidence of success, never fabricate it — but SPAN ALONE cannot tell a
    CUT RECORDING from a SHORT FLIGHT, and grading the second as the first
    deletes real results. A leg is INDETERMINATE only when it did not pass,
    its span is too short to grade, AND its capture is provably incomplete
    (`capture_complete`, resolved at the writer; fail-closed False when the
    terminal time is unreadable). A short leg whose capture reached the
    runtime's own terminal time is a flight OUTCOME and IS graded. Passed
    legs are never excluded. Legs without a span (v1/gap paths) are
    untouched: the gap path already fails closed on its own reason."""
    kept, excluded = [], []
    for r in rows:
        span = r.get("capture_span_s")
        if (span is not None and span < MIN_CAPTURE_S and not r["passed"]
                and not r.get("capture_complete")):
            excluded.append(r)
        else:
            kept.append(r)
    return kept, excluded


def short_problem(excluded):
    return ("INDETERMINATE_SHORT_CAPTURE:"
            + ",".join(r["leg"] for r in excluded))
