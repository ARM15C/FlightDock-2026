#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: FROZEN-SEED CUSTODY for the curriculum-v3 gate relay
#   (T-157, operator-authorized STRUCTURAL extraction from gate_relay_core.py,
#   2026-07-31). Draws fresh seeds from the frozen SEED_BLOCKS order against
#   the append-only campaign ledger — the cursor IS the ledger sum, never a
#   separate counter, never rewound. This is the evidence-chain spine for the
#   H2/H3 comparisons: which seeds flew, in which role, is reconstructable
#   from this file's output alone.
#   Extracted verbatim (pure move, zero behavior change) because core.py sat
#   at its 300-line hard cap and seed custody deserves its own charter; the
#   T-156/R-8 single-writer lock lands HERE as the follow-on BEHAVIORAL
#   dispatch.
# PUBLIC API: next_seeds. (gate_relay_core re-exports it, so every existing
#   caller — process_gate, ws1r1_confirm.sh, tests — is untouched.)
# SIDE-EFFECTS: appends ONE JSONL row to the campaign seed ledger per call.
#   Never deletes, never rewrites, never touches SEED_BLOCKS.json.
# CAPS: file soft 50 / hard 80 LoC (code lines, house convention); function
#   cap 30.
# COVERAGE: tests/unit/test_gate_relay_core.py (cursor append / no reuse /
#   exhaustion — exercised via the core re-export, unchanged);
#   tests/unit/test_gate_relay_seeds.py (T-158 single-writer detectors).
# =============================================================================
import fcntl
import json
import os


def _ledger_rows(ledger_path):
    if not os.path.isfile(ledger_path):
        return []
    with open(ledger_path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def next_seeds(ledger_path, block, n, purpose, seed_blocks_path):
    """Draw n FRESH seeds from the frozen order (cursor = total already
    consumed from this block, per the append-only campaign ledger — never
    restored, never rewound). Appends the usage record BEFORE returning.

    T-158/R-8: the ENTIRE read -> compute -> append is one transaction under
    an exclusive flock on the ledger file itself. Unlocked, two interleaved
    callers computed the same cursor and were handed the IDENTICAL slice
    (double-spend the exhaustion check cannot see); a second writer process
    HAS occurred live (ws1r1_confirm.sh, 07-31). flock excludes across
    processes AND across separate fds in one process; it releases on close,
    so the SystemExit path cannot wedge the ledger. Deliberately NO disable
    flag. Known delta, disclosed: opening "a" before the exhaustion check
    creates an empty ledger file where the old code left none."""
    order = json.load(open(seed_blocks_path))["blocks"][block]["frozen_order"]
    with open(ledger_path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        used = sum(r["n"] for r in _ledger_rows(ledger_path)
                   if r.get("block") == block)
        if used + n > len(order):
            raise SystemExit(
                f"seed block '{block}' exhausted ({used}+{n}>{len(order)})")
        seeds = order[used:used + n]
        fh.write(json.dumps({"block": block, "cursor_before": used, "n": n,
                             "purpose": purpose, "seeds": seeds}) + "\n")
        fh.flush()                  # visible BEFORE the lock releases at close
    return seeds


def pinned_seeds(ledger_path, seeds, n, purpose, seed_blocks_path,
                 block="full_gate"):
    """Re-fly an EXACT prior panel on ITS OWN seeds (T-363, signed prereg
    D1A003-COMPRESSED-20260809 §1). Records the re-use with **n=0** so the
    cursor does NOT advance: a re-fly re-reads custody, it never spends it.
    The seed ORDER is the caller's and is preserved — global_traj_map keys
    trajectory off position, so order IS the pairing with the prior panel.

    Fails CLOSED (SystemExit, the next_seeds convention) on wrong count,
    duplicates, or any seed outside the frozen order — a re-fly that quietly
    flew a different panel is exactly the failure this exists to prevent."""
    order = json.load(open(seed_blocks_path))["blocks"][block]["frozen_order"]
    seeds = [int(s) for s in seeds]
    bad = [s for s in seeds if s not in order]
    if len(seeds) != n or len(set(seeds)) != n or bad:
        raise SystemExit(f"pinned seeds invalid: got {len(seeds)} (need {n} "
                         f"distinct) outside-'{block}'={bad}")
    with open(ledger_path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        used = sum(r["n"] for r in _ledger_rows(ledger_path)
                   if r.get("block") == block)
        fh.write(json.dumps({"block": block, "cursor_before": used, "n": 0,
                             "purpose": purpose, "seeds": seeds,
                             "reuse": True}) + "\n")
        fh.flush()
    return seeds
