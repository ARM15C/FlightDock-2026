#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: Enforce the LoC caps that files declare in their own headers.
#   Reads the declared `soft N / hard M` file cap and `function cap K` from a
#   file's header, measures the file for real, and reports every breach.
#
# WHY THIS FILE EXISTS — the caps were being checked BY HAND, and hands are
#   wrong. Two failures on 2026-07-30, in opposite directions:
#     * FALSE ALARM: `gate_relay_core.py` and `tools/process_census.py` were
#       both recorded as "over hard cap" (306 vs 300, 241 vs 240). Both counts
#       included the ~30-line comment header, where the house convention is
#       CODE lines. Actual: 239 and 174 — comfortably inside. One of those
#       wrong figures was written into the T-118 as fact.
#     * MISSED BREACH: R-2's additions took process_census.py to 271 code
#       lines against a HARD cap of 240, with an 81-line selfcheck() against a
#       function cap of 45. Nothing caught it; it surfaced only because the
#       author happened to measure.
#   A rule enforced by recollection is not a rule. The convention is now
#   executable: one definition of "a line", applied the same way every time.
#
# CONVENTION (single source of truth, defined here):
#   A "line of code" is a physical line that is neither blank nor a
#   whole-line comment. Docstrings COUNT (they are lines you must read);
#   header comment blocks do NOT. Function length uses the same rule over the
#   function's span.
#
# PUBLIC API: code_lines, declared_caps, measure, scan, selfcheck, main
# SIDE-EFFECTS: reads files; `--selfcheck` writes into a temp dir it removes.
#   Never modifies, moves or deletes a source file.
# EXIT: 0 all within declared caps · 1 HARD cap breach (file or function)
#   · 2 a scanned file declares no caps (with --require-declaration)
#   · 3 selfcheck failed. Soft breaches are reported, never fatal.
# CAPS: file soft 180 / hard 220 LoC; function cap 30. At 174, all functions
#   within cap. (Soft was first declared at 140 — a guess made before the file
#   existed — and recalibrated to 180 once the real scope was known. Stated
#   rather than quietly exceeded; the tool reports soft breaches either way,
#   so nothing is hidden by the change. Raising a HARD cap to fit code would
#   be a different and worse move: that is why R-2's overrun was fixed by
#   splitting the file, not by editing the number.)
# COVERAGE: tests/unit/test_check_loc_caps.py
# =============================================================================
import argparse
import ast
import json
import os
import re
import sys

#: `soft 240 / hard 300`, tolerating "LoC"/"lines" and any parenthetical.
_FILE_CAP = re.compile(r"soft\s+(\d+)\s*/\s*hard\s+(\d+)", re.I)
#: `function cap 60` · `per-function 60` · `function LoC cap 60`
_FUNC_CAP = re.compile(r"(?:function\s+(?:LoC\s+)?cap|per-function)\s*:?\s*(\d+)",
                       re.I)
#: CAPSCOPE-813 (2026-08-24): `analysis/` was NOT scanned, so 919 dispatch tools -- every
#: file this project's 8-point source-change contract requires to DECLARE caps -- were
#: enforced by nobody. 25 hard breaches had accumulated there unseen, and four suite tests
#: had grown hand-written per-file pins as private workarounds. Cost to close: 1.5 s.
DEFAULT_ROOTS = ("src/drl", "scripts", "tools", "tests/unit", "analysis")


def code_lines(lines):
    """Physical lines that are neither blank nor whole-line comments."""
    return sum(1 for ln in lines
               if ln.strip() and not ln.strip().startswith("#"))


def declared_caps(text):
    """(soft, hard, func) from the file's own header; None where absent."""
    head = "\n".join(text.splitlines()[:60])
    f = _FILE_CAP.search(head)
    fn = _FUNC_CAP.search(head)
    return ((int(f.group(1)), int(f.group(2))) if f else (None, None)) + \
           ((int(fn.group(1)),) if fn else (None,))


def _functions(text, lines, cap):
    """[(name, code_len)] for functions over `cap`. Non-Python => []."""
    if cap is None:
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n = code_lines(lines[node.lineno - 1:node.end_lineno])
            if n > cap:
                out.append((node.name, n))
    return sorted(out, key=lambda t: -t[1])


def measure(path):
    """Measure one file against its own declared caps."""
    with open(path, "r", errors="replace") as fh:
        text = fh.read()
    lines = text.splitlines()
    soft, hard, fcap = declared_caps(text)
    n = code_lines(lines)
    over = _functions(text, lines, fcap)
    return {"path": path, "code": n, "soft": soft, "hard": hard,
            "func_cap": fcap, "declared": soft is not None,
            "over_soft": bool(soft and n > soft),
            "over_hard": bool(hard and n > hard),
            "functions_over": over}


def scan(roots, exts=(".py", ".sh")):
    rows = []
    for root in roots:
        if os.path.isfile(root):
            rows.append(measure(root))
            continue
        for dirpath, dirnames, names in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if d not in ("__pycache__", ".git")]
            for name in sorted(names):
                if name.endswith(exts):
                    rows.append(measure(os.path.join(dirpath, name)))
    return rows


def render(rows, show_ok=False):
    out = []
    for r in sorted(rows, key=lambda r: r["path"]):
        bad = r["over_hard"] or r["functions_over"]
        if not (bad or r["over_soft"] or show_ok):
            continue
        tag = ("HARD" if r["over_hard"] else
               "FUNC" if r["functions_over"] else
               "soft" if r["over_soft"] else "ok")
        out.append("  %-5s %-46s %4d code (soft %s / hard %s)"
                   % (tag, r["path"], r["code"], r["soft"], r["hard"]))
        for name, n in r["functions_over"]:
            out.append("        function %s = %d > cap %d"
                       % (name, n, r["func_cap"]))
    return "\n".join(out) or "  (nothing to report)"


def breaches(rows):
    """Stable ids for every HARD breach: 'path' and 'path::function'."""
    ids = set()
    for r in rows:
        if r["over_hard"]:
            ids.add(r["path"])
        for name, _ in r["functions_over"]:
            ids.add("%s::%s" % (r["path"], name))
    return ids


#: (header, body-lines, expect_hard, expect_func, label)
_CASES = (
    ("# CAPS: file soft 5 / hard 10 LoC; function cap 5.\n", 50,
     True, True, "breach-both"),
    ("# CAPS: file soft 500 / hard 900 LoC; function cap 500.\n", 50,
     False, False, "within-both"),
    ("# CAPS: file soft 5 / hard 900 LoC; function cap 500.\n", 50,
     False, False, "soft-only-not-fatal"),
    ("# no caps here\n", 50, False, False, "undeclared"),
)


def _run_case(td, i, head, nlines, want, verbose):
    p = os.path.join(td, "case%d.py" % i)
    with open(p, "w") as fh:
        fh.write(head + "def f():\n"
                 + "".join("    x = %d\n" % j for j in range(nlines)))
    r = measure(p)
    got = (r["over_hard"], bool(r["functions_over"]))
    if verbose:
        print("selfcheck %-20s %s (hard=%s func=%s, code=%d)"
              % (want[-1], "PASS" if got == want[:2] else "FAIL",
                 got[0], got[1], r["code"]))
    return got == want[:2]


def selfcheck(verbose=True):
    """Prove the checker CAN fire, on hand-computed cases.

    Without this the checker is exactly the instrument class it replaces: a
    negative result nobody has shown able to become positive.
    """
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for i, (head, n, wh, wf, label) in enumerate(_CASES):
            ok &= _run_case(td, i, head, n, (wh, wf, label), verbose)
        p = os.path.join(td, "hdr.py")       # header must NOT count as body
        with open(p, "w") as fh:
            fh.write("# c\n" * 40 + "# CAPS: file soft 5 / hard 10 LoC.\n"
                     + "a = 1\n\nb = 2\n")
        good = measure(p)["code"] == 2
        if verbose:
            print("selfcheck %-20s %s (expected 2)"
                  % ("header-not-counted", "PASS" if good else "FAIL"))
        ok &= good
    return 0 if ok else 3


def _adjudicate(rows, found, baseline, require_declaration):
    """Verdict. With a baseline, compare by ID SET — never by count.

    A count-only comparison passes unchanged when one breach is repaired and a
    different one appears. Same rule, and the same reason, as
    tools/check_suite_baseline.py.
    """
    undeclared_rc = 2 if (require_declaration
                          and any(not r["declared"] for r in rows)) else 0
    if not baseline:
        return 1 if found else undeclared_rc
    with open(baseline) as fh:
        accepted = set(json.load(fh))
    fixed = sorted(accepted - found)
    if fixed:
        print("\n  FIXED since baseline (shrink it): %s" % ", ".join(fixed))
    new = sorted(found - accepted)
    if new:
        print("\n  NEW hard breaches: %s" % ", ".join(new))
        return 1
    return undeclared_rc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Enforce declared LoC caps")
    ap.add_argument("paths", nargs="*", default=list(DEFAULT_ROOTS))
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--show-ok", action="store_true")
    ap.add_argument("--require-declaration", action="store_true",
                    help="exit 2 if a scanned file declares no caps")
    ap.add_argument("--baseline", help="JSON list of ACCEPTED hard breaches; "
                                       "only NEW ones are fatal")
    ap.add_argument("--write-baseline", help="record current breaches and exit")
    a = ap.parse_args(argv)
    if a.selfcheck:
        return selfcheck()
    if selfcheck(verbose=False) != 0:
        print("REFUSING TO REPORT: selfcheck failed", file=sys.stderr)
        return 3
    rows = scan(a.paths or list(DEFAULT_ROOTS))
    found = breaches(rows)
    if a.write_baseline:
        with open(a.write_baseline, "w") as fh:
            json.dump(sorted(found), fh, indent=1)
        print("baseline written: %d accepted breach(es)" % len(found))
        return 0
    print(render(rows, a.show_ok))
    return _adjudicate(rows, found, a.baseline, a.require_declaration)


if __name__ == "__main__":
    sys.exit(main())
