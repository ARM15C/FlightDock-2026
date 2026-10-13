#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: The live self-test lanes for tools/process_census.py. Each
#   lane spawns a short-lived DECOY and asserts the census reacts the way it
#   must, so every census run proves its matchers can actually fire before any
#   count is trusted.
#
# WHY THIS FILE EXISTS — two reasons, both structural.
#   1. The lanes are scaffolding, not product. Keeping them beside the census
#      logic pushed process_census.py to 271 code lines (hard cap 240) with an
#      81-line selfcheck() (function cap 45). Splitting keeps the instrument
#      itself small enough to audit by eye, which matters for a file whose
#      whole job is being trustworthy.
#   2. Every lane is a POSITIVE or NEGATIVE control for a defect that actually
#      happened on this project — see each lane's docstring. They are
#      regression evidence, and grouping them makes the set legible and hard
#      to quietly shrink.
#
# PUBLIC API: LANES (tuple of (name, callable)), run
# SIDE-EFFECTS: spawns and terminates short-lived `sleep` children that are
#   ALWAYS this process's own and are ALWAYS reaped (self-spawned carve-out).
#   Never signals a process it did not create. Writes one temp file.
# CAPS: file soft 110 / hard 140 LoC (code lines); function cap 20.
# COVERAGE: tests/unit/test_process_census.py
# =============================================================================
import os
import shutil
import tempfile
import time


def _reap(proc):
    proc.terminate()
    proc.wait(timeout=5)


def _matcher_can_fire(pc, root):
    """POSITIVE control. The direct antidote to the 2026-07-30 grep census,
    which reported 0 for a live process for hours because its pattern could
    never match."""
    target = os.path.join(root, pc.CLASSES["fdm"][1])
    base = pc.census(root)["fdm"]["count"]
    proc = pc._decoy(target)
    try:
        time.sleep(0.4)
        live = pc.census(root)["fdm"]["count"]
        return live > base, "%d -> %d" % (base, live)
    finally:
        _reap(proc)


def _relative_argv_resolves(pc, root):
    """POSITIVE control, LIVE-FOUND 2026-07-30 inside the fix itself: the relay
    runs `bash scripts/...` with cwd=FD_ROOT while ssh lands in $HOME, so a
    relative token resolved against OUR cwd never matched a live relay."""
    base = pc.census(root)["fdm"]["count"]
    proc = pc._decoy(pc.CLASSES["fdm"][1], cwd=root)
    try:
        time.sleep(0.4)
        live = pc.census(root)["fdm"]["count"]
        return live > base, "%d -> %d" % (base, live)
    finally:
        _reap(proc)


def _undeclared_can_fire(pc, root):
    """POSITIVE control (R-2). Proves the catch-all fires on a process running
    project code under no declared class — the state in which 11 orphaned
    eval_drl_checkpoint.py ran beside an all-zero census table.

    The probe file is created UNDER root rather than reusing the census tool's
    own path. LIVE-FOUND on the 24xl 2026-07-30: the documented remote
    workflow scp's this tool to /tmp, which is outside FD_ROOT, so a decoy
    carrying `pc.__file__` could not satisfy the predicate and the lane failed
    — on the one box where the census matters most. A self-test must not
    depend on where the tool happens to live.
    """
    parent = os.path.join(root, "logs")          # gitignored when present
    if not os.path.isdir(parent):
        parent = root
    td = tempfile.mkdtemp(dir=parent, prefix=".census_selfcheck_")
    try:
        stray = os.path.join(td, "undeclared_probe.py")
        with open(stray, "w") as fh:
            fh.write("# selfcheck probe: under FD_ROOT, matches no class\n")
        base = len(pc.undeclared(root))
        proc = pc._decoy(stray)
        try:
            time.sleep(0.4)
            rows = pc.undeclared(root)
            # T-166: PID MEMBERSHIP ONLY. A `len(rows) > base` conjunct used to
            # guard this and made the lane fail whenever any unrelated
            # UNDECLARED process exited between the two snapshots: the count
            # cancelled the decoy's +1 while the decoy sat plainly in `rows`.
            # LIVE FLAKE 2026-07-31 — my own gate watcher's 5-minute ssh poll
            # did exactly that, reddening the suite on a healthy box. Counting
            # where identity is in hand is the same count-vs-ID-set error this
            # project already pinned in tools/check_suite_baseline.py.
            hit = any(r["pid"] == proc.pid for r in rows)
            return hit, "%d -> %d" % (base, len(rows))
        finally:
            _reap(proc)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _undeclared_ignores_cwd_only(pc, root):
    """NEGATIVE control (R-2). A process whose cwd is FD_ROOT but which runs no
    project file — a login shell, an editor, an agent session — must NOT be
    reported, or real orphans drown in routine noise."""
    proc = pc._decoy("no-fd-file-here", cwd=root)
    try:
        time.sleep(0.4)
        quiet = not any(r["pid"] == proc.pid for r in pc.undeclared(root))
        return quiet, "cwd-only decoy ignored"
    finally:
        _reap(proc)


def _identity_not_substring(pc, root):
    """NEGATIVE control. Same basename, different inode must NOT match — this
    is what makes a mistyped path a LOUD missing file instead of a silent
    non-match, and what stops a command that merely mentions a name from
    inflating its own count."""
    target = os.path.join(root, pc.CLASSES["fdm"][1])
    base = pc.census(root)["fdm"]["count"]
    with tempfile.TemporaryDirectory() as td:
        near = os.path.join(td, os.path.basename(target))
        with open(near, "w") as fh:
            fh.write("# near-miss: same NAME, different inode\n")
        proc = pc._decoy(near)
        try:
            time.sleep(0.4)
            after = pc.census(root)["fdm"]["count"]
            return after == base, "%d -> %d" % (base, after)
        finally:
            _reap(proc)


#: Order matters only for readability; lanes are independent.
LANES = (
    ("matcher-can-fire", _matcher_can_fire),
    ("relative-argv-resolves", _relative_argv_resolves),
    ("undeclared-can-fire", _undeclared_can_fire),
    ("undeclared-ignores-cwd-only", _undeclared_ignores_cwd_only),
    ("identity-not-substring", _identity_not_substring),
)


def run(pc, root, verbose=True):
    """Run every lane. Returns True only if ALL pass (fail-closed)."""
    ok = True
    for name, fn in LANES:
        passed, detail = fn(pc, root)
        if verbose:
            print("selfcheck %s: %s (%s)"
                  % (name, "PASS" if passed else "FAIL", detail))
        ok &= bool(passed)
    return ok
