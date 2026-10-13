#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: Terminal-state and attempt-directory bookkeeping for the
#   curriculum-v3 gate relay. Supplies the two invariants the relay lacked:
#     (1) every gate attempt gets its OWN leg directory, so a retry can never
#         pool its legs with a prior attempt's;
#     (2) every TERMINAL outcome — success AND every failure rc — is recorded,
#         so the poller cannot re-dispatch a request that already finished.
#
# WHY THIS FILE EXISTS — a retry loop that poisoned its own input.
#   2026-07-30, WS1-r0's W2 gate. `gate_relay_core.process_gate()` built its
#   leg directory as `gate_{label}` with exist_ok=True, and wrote the
#   `.processed` marker ONLY after publishing a verdict. Every failure rc
#   (2/3/4/6) returned with NO marker, so `curriculum_gate_relay.sh` re-
#   dispatched the same request into the SAME directory. The retention screen
#   then counted legs from ALL attempts, mismatched its expected N, and
#   returned screen_adjudication_invalid => rc=6 RETENTION HOLD => retry =>
#   more legs in the same directory. Self-reinforcing: it could never
#   converge, because each attempt made the next attempt's input strictly
#   worse. Observed end state: 36 leg directories in one `gate_W2/`
#   (3 attempts x 12) and 84 eval seeds burned — seeds are drawn BEFORE the
#   spawn and are never rewound (append-only ledger, by design).
#
#   The fix is two-sided ON PURPOSE. Per-attempt directories alone would stop
#   the pooling but still burn a fresh seed block on every silent retry;
#   terminal markers alone would stop the retry but leave already-pooled trees
#   ambiguous. Together: an attempt is isolated, and a failure PAGES A HUMAN
#   instead of spending seeds. Training pausing is the safe state — the
#   curriculum callback simply waits for a verdict that never arrives.
#
# PUBLIC API: attempt_dir, terminal_state, hold_sentinel, read_terminal,
#   is_terminal, TERMINAL_SUFFIX
# SIDE-EFFECTS: creates one attempt directory under the caller's gate root;
#   writes ONE terminal-marker file next to the request (atomic tmp+replace);
#   on a HOLD, writes ONE GATE_HOLD_<label>.json into the LEARNER's log_dir
#   (= dirname(verdicts_dir), never inside verdicts_dir — T-237/B2).
#   Never deletes, never signals a process, never touches a published verdict.
# CAPS: file soft 70 / hard 90 LoC (code lines, excl. comments/blanks; the
#   house convention — see gate_relay_core.py); function cap 25. At 63
#   (T-237 added hold_sentinel, 11 code lines); still under SOFT.
# COVERAGE: tests/unit/test_gate_relay_state.py
# =============================================================================
import json
import os

TERMINAL_SUFFIX = ".terminal"
_PREFIX = "attempt_"


def attempt_dir(gate_root):
    """Create and return (path, k) for a FRESH attempt directory.

    k is derived from the attempt directories that ALREADY EXIST — never from
    a clock, a counter file, or a random draw. Two reasons: the relay must be
    reconstructable from a leg tree alone (an attempt's number is a property
    of the tree, not of when it ran), and a clock-derived name cannot be
    asserted in a test.
    """
    os.makedirs(gate_root, exist_ok=True)
    used = [int(n[len(_PREFIX):]) for n in os.listdir(gate_root)
            if n.startswith(_PREFIX) and n[len(_PREFIX):].isdigit()]
    k = max(used, default=0) + 1
    path = os.path.join(gate_root, "%s%03d" % (_PREFIX, k))
    # Deliberately NOT exist_ok: a collision means two relays are racing the
    # same gate root, which must be LOUD, never silently shared (that sharing
    # is precisely the 07-30 defect).
    os.makedirs(path)
    return path, k


def terminal_state(request_path, rc, reason, **extra):
    """Record a TERMINAL outcome for this request; return rc unchanged.

    Written for EVERY terminal rc, success included. The poller treats the
    marker's PRESENCE as "do not re-dispatch" regardless of rc, so a HOLD
    stays a HOLD until a human removes it.

    Not written for dry runs — a rehearsal that marked the request terminal
    would block the real gate that follows it.
    """
    rec = {"rc": int(rc), "reason": str(reason)}
    rec.update(extra)
    tmp = request_path + TERMINAL_SUFFIX + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True, default=str)
    os.replace(tmp, request_path + TERMINAL_SUFFIX)     # atomic publish
    return rc


def hold_sentinel(verdicts_dir, label, rc, reason, **extra):
    """Tell the LEARNER that this gate is DEAD, not merely late (T-237/B2).

    A HOLD writes RETENTION_HALT_*.json into legs_dir and .terminal next to
    the request — neither of which the learner can see. Its pause predicate is
    "the published verdict does not exist yet", so a HOLD was indistinguish-
    able from a slow panel: it waited out the 300-min fail-open and then
    trained to cap (measured: 45,738 un-gated steps, WS34 stage 12).

    Written to the learner's OWN log_dir = dirname(verdicts_dir), NEVER into
    verdicts_dir itself — the ack writer globs *.json there.
    """
    log_dir = os.path.dirname(os.path.abspath(str(verdicts_dir).rstrip("/")))
    rec = {"label": str(label), "rc": int(rc), "reason": str(reason)}
    rec.update(extra)
    path = os.path.join(log_dir, "GATE_HOLD_%s.json" % label)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)                                # atomic publish
    return path


def read_terminal(request_path):
    """Marker contents, or None if unreadable/absent.

    None is NOT authoritative for control flow — use is_terminal() for that.
    """
    try:
        with open(request_path + TERMINAL_SUFFIX) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def is_terminal(request_path):
    """True if this request reached a terminal state and must NOT re-dispatch.

    Keyed on file EXISTENCE, not on parseable contents: a truncated or corrupt
    marker still blocks re-dispatch. Defining this as
    `read_terminal(...) is not None` would make a corrupt marker mean "retry",
    reopening the infinite-retry path this module exists to close.
    """
    return os.path.exists(request_path + TERMINAL_SUFFIX)
