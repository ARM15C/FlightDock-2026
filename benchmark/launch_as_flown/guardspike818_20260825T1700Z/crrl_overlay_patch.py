#!/usr/bin/env python3
# RESPONSIBILITY: the DECLARED-LINE PATCH LAYER for the CRRL overlay exam, and nothing else. It
#   owns the two declared line pairs (the alt-NEG floor, and -- when a beta overlay is asked for --
#   CRRL_BPC_BETA), applies them, proves the working tree carries EXACTLY the declared edit and no
#   other, and reverts with a witness.
#   SPLIT OUT of crrl_overlay_exam.py on 2026-08-28 when that file breached its hard cap after the
#   two-lever change: CAPS ARE SPLIT, NEVER RAISED. The launcher re-exports these names, so every
#   existing caller and every existing selftest limb keeps working unchanged.
# WHY A NEAR-MISS MUST REFUSE: the declared OLD text must appear EXACTLY once. A line differing
#   only in whitespace is a DIFFERENT line, and patching it would edit a tree we do not recognise.
# PUBLIC API: sh(*a) -> str; declared_new(dose) -> str; declared_new_beta(beta) -> str;
#   apply_patch(dose, path=None, beta=None) -> None; assert_only_declared(dose, beta=None) -> str;
#   revert_patch() -> str. Constants: FD, PATCH_FILE, OLD, NEW, OLD_BETA, NEW_BETA.
# SIDE-EFFECTS: rewrites and reverts one file (src/drl/config.py) in the working tree; runs git
#   read commands. Spawns no exam, draws no seed, trains nothing.
# NOT A TRUTH PATH: imports nothing from src/; never reads telemetry; never touches a checkpoint.
# SIZE: soft 70 / hard 90 code-lines; function cap 30. The declared line-pairs were
#   SPLIT OUT to crrl_overlay_levers.py (2026-09-01) when the knob levers took this
#   file to 91 against the hard 90: CAPS ARE SPLIT, NEVER RAISED.
# COVERAGE: crrl_overlay_exam_selftest.py O1/O2/O4/O9/O10/O11/O12 (executed, driven both ways);
#   the KNOBGUARD pre-patch refusal by crrl_overlay_limbs_gamma.py G1-G4 (both directions).
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FD = os.environ.get("FD_ROOT", "/home/ubuntu/FlightDock")
PATCH_FILE = "src/drl/config.py"
from crrl_overlay_levers import (   # noqa: E402 re-exported for every caller
    OLD, NEW, OLD_BETA, NEW_BETA, OLD_CLAMP, NEW_CLAMP, OLD_BIAS, NEW_BIAS,
    OLD_VOFF, NEW_VOFF,
    declared_new, declared_new_beta, declared_new_clamp, declared_new_bias,
    declared_new_voff, _pairs)


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


def _swap(txt, old, new, path):
    if txt.count(old) != 1:
        raise RuntimeError("the declared OLD line is not present exactly once in %s -- refusing "
                           "to patch a tree I do not recognise" % path)
    return txt.replace(old, new)



def _assert_config_clean(p):
    """KNOBGUARD-20260901: REFUSE to patch a config.py ALREADY dirty vs HEAD. Every exam
    refusal path runs `finally: revert_patch()` -- `git checkout --`, which WIPES uncommitted
    edits (burned once, 2026-08-28) -- so a dirty tree refuses BEFORE the write, while refusing
    is free. Temp-file callers (selftests) are exempt by path identity, not convention."""
    if os.path.realpath(p) != os.path.realpath(os.path.join(FD, PATCH_FILE)):
        return
    if sh("git", "-C", FD, "diff", "--name-only", "HEAD", "--", PATCH_FILE).strip():
        raise RuntimeError("KNOBGUARD: %s is ALREADY dirty vs HEAD -- patching now would let the "
                           "eventual `git checkout` revert DESTROY uncommitted work; commit or "
                           "restore it first" % PATCH_FILE)

def apply_patch(dose, path=None, beta=None, clamp=None, bias=None, voff=None):
    """Rewrite EXACTLY the declared lines -- one per elected lever. Refuses if the tree already
    differs from any declared old line, so a whitespace near-miss is a refusal, not a patch."""
    p = path or os.path.join(FD, PATCH_FILE)
    _assert_config_clean(p)
    txt = open(p).read()
    for old, new in _pairs(dose, beta, clamp, bias, voff):
        txt = _swap(txt, old, new, p)
    open(p, "w").write(txt)


def assert_only_declared(dose, beta=None, clamp=None, bias=None, voff=None):
    """Exactly one file, exactly one line, and it is the declared edit. Analysis output may be
    dirty; SOURCE may not."""
    dirty = [q for q in sh("git", "-C", FD, "diff", "--name-only", "HEAD").splitlines() if q.strip()]
    others = [q for q in dirty if q != PATCH_FILE and not q.startswith("analysis/")]
    if others:
        raise RuntimeError("undeclared SOURCE modifications present, refusing: %s" % others)
    if PATCH_FILE not in dirty:
        raise RuntimeError("the declared patch is NOT applied (dirty: %s)" % (dirty or "nothing"))
    want = [new for _, new in _pairs(dose, beta, clamp, bias, voff)]
    n = str(len(want))
    if sh("git", "-C", FD, "diff", "--numstat", "--", PATCH_FILE).split("\t")[:2] != [n, n]:
        raise RuntimeError("expected exactly %s insertion(s) and %s deletion(s) in %s"
                           % (n, n, PATCH_FILE))
    diff = sh("git", "-C", FD, "diff", "--", PATCH_FILE)
    plus = [l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    if sorted(plus) != sorted(want):
        raise RuntimeError("the working-tree edit is NOT the declared patch: %r" % plus)
    return diff


def revert_patch():
    subprocess.run(["git", "-C", FD, "checkout", "--", PATCH_FILE])
    dirty = [q for q in sh("git", "-C", FD, "diff", "--name-only", "HEAD").splitlines()
             if q.strip() and not q.startswith("analysis/")]
    if dirty:
        raise RuntimeError("REVERT FAILED: source still dirty: %s" % dirty)
    return "REVERT WITNESS: no src/ entry in `git diff --name-only HEAD`"
