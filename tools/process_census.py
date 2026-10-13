#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: Authoritative process census for FlightDock campaign classes
#   (SITL, FDM, training, relay, leader, mavproxy, eval spawner).
#
# WHY THIS FILE EXISTS — a census pattern that CANNOT MATCH reports 0 forever.
#   On 2026-07-30 every census in a long session used `grep -c "drl[.]train"`
#   against `ps` output. The real command line is `src/drl/train.py` — a SLASH,
#   not a dot — so the pattern could never match and reported 0 while training
#   was demonstrably alive (PID 1347026, etime 05:25:55). A check that always
#   returns the same answer is not a check. It is the identical failure class
#   to E-4's inert regression window (F-2) and to frv2's zero-padded witness
#   lanes (F-3): an instrument that cannot fire, reporting silence as safety.
#
# THE FIX IS STRUCTURAL, NOT A CORRECTED REGEX. A regex can be mistyped again.
#   1. Identity is FILESYSTEM identity — `os.path.samefile` against a canonical
#      path (argv entry, or /proc/<pid>/exe for compiled binaries). A typo
#      becomes a MISSING FILE, which is loud, instead of a non-matching string,
#      which is silent. It also cannot self-match a shell command that merely
#      mentions the name (the 07-30 phantom-count class).
#   2. A class whose canonical target does not exist on disk reports
#      **UNVERIFIABLE**, never 0.
#   3. `--selfcheck` spawns a LIVE DECOY that the matcher MUST find, and a
#      near-miss decoy it MUST NOT. This proves, every run, that the matcher
#      can actually fire — the exact property the old pattern silently lacked.
#      Decoys are this process's own children and are always reaped.
#
#   R-2 (2026-07-30) — A CLASS CENSUS CAN ONLY ANSWER ABOUT ITS OWN CLASSES.
#   Both structural properties above were working, every count was correct,
#   and the table still read "clean" while 11 orphaned
#   `scripts/eval_drl_checkpoint.py` processes were live — that class was
#   simply never declared. UNVERIFIABLE does not cover this: it means "a
#   DECLARED target is missing from disk", and says nothing about a process
#   running under no declared class at all. Two fixes: declare the missing
#   class, and add `undeclared()` — a catch-all reporting anything executing
#   project code that matches no class, so the next unknown is caught without
#   having been predicted.
#
#   T-134 (2026-07-31) — A GATE THAT CANNOT PASS IS ALSO WORTHLESS.
#   `undeclared()` excluded only `os.getpid()`, the PYTHON census process,
#   while it flags any process carrying a project file in argv. So
#   `bash <driver-under-root>` ALWAYS matched, and gate G7 of the SIGNED
#   WS1-r1 launch driver reported the driver's own shell as an orphan
#   (pid 55521, `files [ws1r1_launch.sh]`, 3 s before its own census) on a
#   demonstrably clean box. Mirror image of the cannot-fire class above: this
#   one failed CLOSED, so it blocked an authorised launch instead of waving
#   through a dirty box — but it could never have passed. The exclusion is now
#   the invoking ANCESTOR CHAIN (`ancestry()`), scoped to this catch-all only
#   and DISCLOSED in the output, so it can never become a silent amnesty.
#
# PUBLIC API: census, undeclared, ancestry, selfcheck, render, main
# SIDE-EFFECTS: reads /proc; `--selfcheck` spawns and terminates its OWN
#   short-lived `sleep` children (self-spawned carve-out, always reaped, and
#   reported). Never signals any process it did not create. Never kills a
#   campaign process. Never deletes anything.
# EXIT: 0 clean · 1 live campaign processes found (with --fail-nonzero)
#   · 2 UNVERIFIABLE class present (fail-closed) · 3 selfcheck failed
#   · 4 UNDECLARED process present (fail-closed: unknown is not clean).
# CAPS: file soft 200 / hard 240 LoC (code lines, excl. comments/blanks — the
#   house convention); function cap 45. At 211: over soft, under hard. The
#   self-test lanes were moved to process_census_lanes.py to keep this file
#   auditable rather than raise the cap. T-134 took it to 239: still UNDER the
#   hard cap, which was NOT raised — the addition was tightened to fit (the
#   T-125 rule). At the next growth this file must SPLIT along the
#   identity/policy seam (argv-exe-ppid matchers out, CLASSES + census +
#   undeclared + main stay). NOTE: the T-118 called
#   this file "241 LoC vs hard cap 240 — one over"; that counted TOTAL lines
#   including the header. On the code-line convention it was 174, well within.
# COVERAGE: tests/unit/test_process_census.py
# =============================================================================
import argparse
import json
import os
import subprocess
import sys
import time
import types

FD = os.environ.get("FD_ROOT", "/home/ubuntu/FlightDock")
UNVERIFIABLE = "UNVERIFIABLE"
UNDECLARED = "UNDECLARED"

#: class -> (match kind, canonical path). "exe" matches /proc/<pid>/exe;
#: "argv" matches any argv entry by samefile.
CLASSES = {
    "sitl":          ("exe",  "ardupilot_formation/build/sitl/bin/arduplane"),
    "fdm":           ("argv", "tools/json_fdm/plane6dof_fdm.py"),
    "training":      ("argv", "src/drl/train.py"),
    "leader":        ("argv", "tests/leader_broadcast_fixed.py"),
    "relay":         ("argv", "scripts/curriculum_gate_relay.sh"),
    "relay_core":    ("argv", "scripts/gate_relay_core.py"),
    "eval_spawner":  ("argv", "scripts/curriculum_eval_spawner.py"),
    "sitl_starter":  ("argv", "tests/start_formation_sitl_fixed.sh"),
    "mavproxy":      ("argv", "flightdock_native_venv/bin/mavproxy.py"),
    # R-2: 11 orphans of this ran undetected on 2026-07-30 — it was simply
    # never declared, and an undeclared class is invisible to a class census.
    "eval_checkpoint": ("argv", "scripts/eval_drl_checkpoint.py"),
}


def _argv(pid, proc="/proc"):
    try:
        with open("%s/%d/cmdline" % (proc, pid), "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    return [p.decode("utf-8", "replace") for p in raw.split(b"\x00") if p]


def _exe(pid, proc="/proc"):
    try:
        return os.readlink("%s/%d/exe" % (proc, pid))
    except OSError:
        return None


def _same(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _pids(proc="/proc"):
    try:
        return sorted(int(n) for n in os.listdir(proc) if n.isdigit())
    except OSError:
        return []


def _candidates(pid, tok, proc):
    """A relative argv token resolves against the TARGET's cwd, not ours.

    Found live 2026-07-30: the relay runs as `bash scripts/curriculum_gate_relay.sh`
    with cwd=/home/ubuntu/FlightDock. Resolving that token against the census
    process's own cwd (ssh lands in $HOME) never matched, so a LIVE relay
    counted 0 — the same cannot-fire class this module exists to prevent.
    """
    if os.path.isabs(tok):
        return [tok]
    return [tok, os.path.join("%s/%d/cwd" % (proc, pid), tok)]


def _matches(pid, kind, target, proc="/proc"):
    if kind == "exe":
        exe = _exe(pid, proc)
        return bool(exe) and _same(exe, target)
    argv = _argv(pid, proc)
    if not argv:
        return False
    return any(_same(cand, target)
               for tok in argv for cand in _candidates(pid, tok, proc))


def census(root=FD, classes=None, proc="/proc"):
    """Count live processes per class. Missing target => UNVERIFIABLE, not 0."""
    classes = CLASSES if classes is None else classes
    me = os.getpid()
    out = {}
    for name, (kind, rel) in sorted(classes.items()):
        target = os.path.join(root, rel)
        if not os.path.exists(target):
            out[name] = {"state": UNVERIFIABLE, "target": target,
                         "reason": "canonical target absent — this class "
                                   "CANNOT be detected; refusing to report 0"}
            continue
        hits = [p for p in _pids(proc)
                if p != me and _matches(p, kind, target, proc)]
        out[name] = {"state": "OK", "target": target,
                     "count": len(hits), "pids": hits}
    return out


def _started(pid, proc="/proc"):
    """Process start time (mtime of /proc/<pid>), ISO-Z. None if it exited."""
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ",
                             time.gmtime(os.stat("%s/%d" % (proc, pid)).st_mtime))
    except OSError:
        return None


def _ppid(pid, proc="/proc"):
    try:
        with open("%s/%d/status" % (proc, pid)) as fh:
            for ln in fh:
                if ln.startswith("PPid:"):
                    return int(ln.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def ancestry(pid, proc="/proc", limit=64):
    """This invocation's own chain: `pid` plus its parents, up to init.

    T-134: `undeclared()` used to exclude only `os.getpid()` — the PYTHON
    census process — while `_fd_files_in_argv` flags anything carrying a
    project file in argv. `bash <driver-under-root>` therefore always matched,
    so gate G7 of the WS1-r1 launch driver could NEVER pass: it reported its
    own shell as an undeclared orphan. Bounded and cycle-safe, because a
    corrupt PPid must fail closed, never hang.
    """
    chain, seen = [], set()
    while pid and pid > 1 and pid not in seen and len(chain) < limit:
        chain.append(pid)
        seen.add(pid)
        pid = _ppid(pid, proc)
    return chain


def _fd_files_in_argv(pid, root, proc="/proc"):
    """Argv tokens resolving to an existing regular FILE under root.

    The discriminator is deliberately "executes project code", NOT "cwd is
    inside the project": a login shell, an editor or an agent session sitting
    in FD_ROOT is not a campaign process, and keying on cwd would bury a real
    orphan in routine noise. A directory argument (e.g. `grep -r x scripts/`)
    is likewise not execution, hence isfile and not exists.
    """
    root = os.path.realpath(root) + os.sep
    hits = []
    for tok in (_argv(pid, proc) or []):
        for cand in _candidates(pid, tok, proc):
            try:
                real = os.path.realpath(cand)
            except OSError:
                continue
            if os.path.isfile(real) and real.startswith(root):
                hits.append(real)
                break
    return hits


def undeclared(root=FD, classes=None, proc="/proc", exclude=None):
    """Live processes running project code that match NO declared class.

    WHY THIS EXISTS — UNVERIFIABLE and UNDECLARED are different holes.
      UNVERIFIABLE means "a DECLARED target is missing from disk, so I cannot
      detect that class". It says nothing whatsoever about a process that is
      running under no declared class at all. On 2026-07-30 that distinction
      cost us: the census printed a clean all-zero table while 11 orphaned
      `scripts/eval_drl_checkpoint.py` processes were live, because that class
      simply was not in CLASSES. Every count was correct and the conclusion
      ("box is clean") was false.

      A class census can only ever answer questions about its own classes.
      This function is the complement: it answers "what is running that I did
      not think to ask about?" — the question that catches the next unknown.
    """
    classes = CLASSES if classes is None else classes
    declared = [(kind, os.path.join(root, rel))
                for kind, rel in classes.values()]
    # T-134: the whole invoking chain, not just `os.getpid()`. Scoped to this
    # catch-all ONLY — `census()` stays blind to who invoked it, so a declared
    # campaign process that happens to be an ancestor is still COUNTED.
    mine = set(ancestry(os.getpid(), proc) if exclude is None else exclude)
    rows = []
    for pid in _pids(proc):
        if pid in mine:               # never self-match (the 07-30 phantom)
            continue
        if any(os.path.exists(t) and _matches(pid, k, t, proc)
               for k, t in declared):
            continue                  # accounted for by a declared class
        files = _fd_files_in_argv(pid, root, proc)
        if files:
            rows.append({"pid": pid, "exe": _exe(pid, proc),
                         "started": _started(pid, proc), "files": files})
    return rows


def _decoy(path, seconds=8, cwd=None):
    """A child that merely CARRIES `path` in its argv — never executes it."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time,sys; time.sleep(float(sys.argv[1]))",
         str(seconds), path],
        cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def selfcheck(root=FD, verbose=True):
    """Prove every matcher CAN fire before any count is trusted.

    Lanes live in process_census_lanes.py — imported HERE, not at module scope,
    so the lanes import nothing from us and no cycle exists. They receive an
    EXPLICIT api object rather than `sys.modules[__name__]`, because the test
    suite loads this file via importlib without registering it in sys.modules.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import process_census_lanes as lanes
    api = types.SimpleNamespace(CLASSES=CLASSES, census=census,
                                undeclared=undeclared, _decoy=_decoy,
                                __file__=os.path.abspath(__file__))
    missing = [n for n, (_, rel) in sorted(CLASSES.items())
               if not os.path.exists(os.path.join(root, rel))]
    if verbose:
        print("selfcheck targets-exist: %s%s"
              % ("PASS" if not missing else "FAIL", " " + ",".join(missing)))
    if not os.path.exists(os.path.join(root, CLASSES["fdm"][1])):
        return 3                      # no anchor target => cannot self-test
    return 0 if (lanes.run(api, root, verbose) and not missing) else 3


def render(rows, extras=None, mine=None):
    lines = ["[CENSUS] %s" % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())]
    for name, r in sorted(rows.items()):
        if r["state"] == UNVERIFIABLE:
            lines.append("  %-16s %s — %s" % (name, UNVERIFIABLE, r["reason"]))
        else:
            lines.append("  %-16s %d%s" % (name, r["count"],
                                           "  pids=%s" % r["pids"]
                                           if r["pids"] else ""))
    # Never a bare count: an undeclared process is only actionable if the
    # operator can identify and attribute it.
    lines.append("  %-16s %d" % (UNDECLARED, len(extras or [])))
    for p in (extras or []):
        lines.append("      pid=%s started=%s exe=%s"
                     % (p["pid"], p["started"], p["exe"]))
        for f in p["files"]:
            lines.append("        %s" % f)
    if mine:
        lines.append("  self-chain excluded: %s" % ",".join(map(str, mine)))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="FlightDock process census")
    ap.add_argument("--root", default=FD)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--fail-nonzero", action="store_true",
                    help="exit 1 if any campaign class has live processes")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--no-selfcheck", action="store_true",
                    help="skip the live decoy (read-only environments)")
    a = ap.parse_args(argv)
    if a.selfcheck:
        return selfcheck(a.root)
    if not a.no_selfcheck and selfcheck(a.root, verbose=False) != 0:
        print("CENSUS REFUSES TO REPORT: selfcheck failed — the matcher could "
              "not be proven able to fire.", file=sys.stderr)
        return 3
    rows = census(a.root)
    # Disclosed, never silent: an exclusion that leaves no trace is the T-131
    # audit-self-erasure defect. The operator can always see what was skipped.
    mine = ancestry(os.getpid())
    extras = undeclared(a.root, exclude=set(mine))
    if a.json:
        print(json.dumps({"classes": rows, "undeclared": extras,
                          "self_ancestry_excluded": mine},
                         indent=1, sort_keys=True))
    else:
        print(render(rows, extras, mine))
    if any(r["state"] == UNVERIFIABLE for r in rows.values()):
        return 2
    if extras:                        # fail-closed: unknown != clean
        return 4
    if a.fail_nonzero and any(r["count"] for r in rows.values()):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
