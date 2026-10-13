#!/usr/bin/env python3
"""
RESPONSIBILITY
    Render the v3 curriculum ladder from `src.drl.config` into ONE generated
    markdown block, and detect when the published copy has gone stale.

    This exists because the ladder was re-typed by hand in prose in several
    places with nothing binding any copy to the code, and the copies silently
    diverged: as of 2026-08-06 `docs/CURRICULUM_ROADMAP.md` carried THREE
    mutually contradictory ladders, two of them in the same section. One fact
    must have one name. The table is therefore GENERATED, never authored, and
    `--check` is the gate that makes a stale copy fail loudly instead of being
    read as fact.

    This module is NOT authoritative about the ladder. `src.drl.config` is.
    Every value here is read from the live config objects; nothing is re-typed.

PUBLIC API
    render() -> str the generated block body (no markers)
    read_block(path) -> str|None the block currently published in `path`
    check(path) -> int 0 in sync - 1 stale/absent (fail-closed)
    write(path) -> int 0; rewrites `path` from config
    main(argv) -> int

SIDE EFFECTS
    --check / --selfcheck: READS ONLY.
    --write: rewrites exactly one file (default docs/V3_LADDER_TABLE.md).
    No process signals. No network. Never writes to config or to any doc it
    was not pointed at.

CAPS
    file soft 195 / hard 220 code-lines - function cap 40 LoC
    (soft RAISED DECLARED 180 -> 195 at 2026-08-06: splitting selfcheck() into
    _file_cases/_config_cases to honour the 40-LoC function cap cost ~10 lines
    of signature and return plumbing. Hard cap unchanged.)
COVERAGE
    --selfcheck, executed, every case asserted in BOTH directions, plus
    tests/unit/test_ladder_doc_in_sync.py which pins the published copy.
"""
import os
import sys
import tempfile

FD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEGIN = "<!-- BEGIN GENERATED: v3-ladder -->"
END = "<!-- END GENERATED: v3-ladder -->"
DEFAULT_OUT = os.path.join(FD, "docs", "V3_LADDER_TABLE.md")

_PREAMBLE = """# V3 CURRICULUM LADDER — GENERATED, DO NOT HAND-EDIT

Source of truth: `src/drl/config.py` (`_V3_LADDER_SPEC` and the tables derived
from it). Regenerate with `python tools/render_ladder_table.py --write`; verify
with `--check` (rc=1 when stale). `tests/unit/test_ladder_doc_in_sync.py` fails
the suite if this file drifts from the code.

Any hand edit inside the generated markers will be overwritten and will fail the
detector first. Prose about the ladder belongs in `docs/CURRICULUM_ROADMAP.md`,
which points here rather than restating the numbers.
"""


def _config():
    if FD not in sys.path:
        sys.path.insert(0, FD)
    from src.drl import config
    return config


def rows():
    """Ladder rows straight off the live config objects. No re-typing."""
    c = _config()
    out = []
    # RERAMP-811: iterate the WHOLE ladder (MAX), not just up to C1. This
    # renderer and the doc-sync test both derived their range from C1, so an
    # append past C1 rendered a short table AND the test still passed --
    # two derivations from one stale bound cannot disagree.
    for i in range(c.MAX_STAGE_INDEX_V3 + 1):
        st = c.CURRICULUM_STAGES_V3[i]
        sig = st.wind_sigma_ms
        out.append({
            "index": i,
            "label": c.STAGE_LABELS_V3[i],
            "wind_ms": float(st.wind_max_ms),
            "sigma": (float(sig[0]) if sig else None),
            "nominal": c.STAGE_NOMINAL_TIMESTEPS_V3[i],
            "cap": c.STAGE_CAP_TIMESTEPS_V3[i],
            "gate": c.V3_GATE_TYPE[i],
            "anchor": i in c.V3_ANCHOR_STAGES,
        })
    return out


def render():
    """The generated block body. Deterministic; no clock, no environment."""
    c = _config()
    rs = rows()
    lines = [
        "| Stage | Label | Wind m/s | σ | Nominal | Cap | Gate | Anchor |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rs:
        lines.append("| %d | %s | %.1f | %s | %s | %s | %s | %s |" % (
            r["index"], r["label"], r["wind_ms"],
            ("—" if r["sigma"] is None else "%.2f" % r["sigma"]),
            "{:,}".format(r["nominal"]), "{:,}".format(r["cap"]),
            r["gate"], ("**ANCHOR**" if r["anchor"] else "")))
    lines += [
        "",
        "- Rungs: **%d** (indices 0–%d) · anchors **%s** · C1 index **%d**"
        % (len(rs), c.MAX_STAGE_INDEX_V3, str(tuple(c.V3_ANCHOR_STAGES)),
           c.C1_STAGE_INDEX_V3),
        "- Learner pool (BINDING + INCLUSIVE): **%s** transitions"
        % "{:,}".format(c.LEARNER_POOL_INCLUSIVE_V3),
        "- Nominal sum **%s** · cap sum **%s** (cap sum is DIAGNOSTIC ONLY — the"
        " pool is the single binding constraint)"
        % ("{:,}".format(sum(r["nominal"] for r in rs)),
           "{:,}".format(sum(r["cap"] for r in rs))),
    ]
    return "\n".join(lines)


def read_block(path):
    """The block currently published in `path`, or None if absent/malformed."""
    if not os.path.isfile(path):
        return None
    text = open(path).read()
    if BEGIN not in text or END not in text:
        return None
    body = text.split(BEGIN, 1)[1].split(END, 1)[0]
    return body.strip("\n")


def check(path=DEFAULT_OUT):
    """rc=0 published block matches config - rc=1 stale or absent."""
    cur = read_block(path)
    want = render()
    if cur is None:
        print("LADDER BLOCK ABSENT or malformed in %s (markers missing)" % path)
        return 1
    if cur != want:
        print("LADDER BLOCK STALE in %s" % path)
        cl, wl = cur.split("\n"), want.split("\n")
        for n in range(max(len(cl), len(wl))):
            a = cl[n] if n < len(cl) else "<missing>"
            b = wl[n] if n < len(wl) else "<missing>"
            if a != b:
                print("  line %d\n    published: %s\n    config   : %s"
                      % (n + 1, a, b))
        return 1
    print("LADDER BLOCK IN SYNC with src/drl/config.py (%s)" % path)
    return 0


def write(path=DEFAULT_OUT):
    """Rewrite `path` from config. The ONLY sanctioned way to change it."""
    with open(path, "w") as fh:
        fh.write("%s\n%s\n%s\n%s\n" % (_PREAMBLE, BEGIN, render(), END))
    print("wrote %s" % path)
    return 0


def _file_cases(tmp, body):
    """Temp-file cases: check() must be able to FAIL and to PASS."""
    good = os.path.join(tmp, "good.md")
    write(good)
    cases = [("freshly written file is IN SYNC", check(good) == 0)]
    open(good, "w").write(
        open(good).read().replace("| 0 | S0 |", "| 0 | XX |"))
    cases.append(("hand-corrupted block is STALE", check(good) == 1))
    bare = os.path.join(tmp, "bare.md")
    open(bare, "w").write("no markers here\n")
    cases.append(("missing markers fail CLOSED", check(bare) == 1))
    cases.append(("absent file fails CLOSED",
                  check(os.path.join(tmp, "nope.md")) == 1))
    rt = os.path.join(tmp, "rt.md")
    write(rt)
    cases.append(("block round-trips: read_block == render",
                  read_block(rt) == body))
    cases.append(("a DIFFERENT body does not round-trip",
                  read_block(rt) != body + "\n| 99 | ZZ |"))
    return cases


def _config_cases(body, rs, c):
    """Derivation cases: every rendered value traced back to config."""
    return [
        ("row count == MAX index + 1", len(rs) == c.MAX_STAGE_INDEX_V3 + 1),
        ("C1 index is present as a row", any(
            r["index"] == c.C1_STAGE_INDEX_V3 for r in rs)),
        ("every anchor index is in the table",
         all(any(r["index"] == a for r in rs) for a in c.V3_ANCHOR_STAGES)),
        ("labels come from config, not this file",
         all(r["label"] == c.STAGE_LABELS_V3[r["index"]] for r in rs)),
        ("sigma read from the config stage object",
         all(r["sigma"] is None or abs(
             r["sigma"] - c.CURRICULUM_STAGES_V3[r["index"]].wind_sigma_ms[0]
         ) < 1e-12 for r in rs)),
        ("render is deterministic", render() == body),
    ]


def selfcheck():
    """Both directions, every case. rc=0 GREEN / rc=1 RED."""
    body = render()
    with tempfile.TemporaryDirectory() as tmp:
        cases = _file_cases(tmp, body)
    cases += _config_cases(body, rows(), _config())
    ok = True
    for name, res in cases:
        print("  %-46s %s" % (name, "PASS" if res else "FAIL"))
        ok = ok and res
    print("SELFCHECK %s (%d cases)" % ("rc=0 GREEN" if ok else "rc=1 RED",
                                       len(cases)))
    return 0 if ok else 1


def main(argv):
    path = (argv[argv.index("--out") + 1] if "--out" in argv else DEFAULT_OUT)
    if "--selfcheck" in argv:
        return selfcheck()
    if "--write" in argv:
        return write(path)
    if "--check" in argv:
        return check(path)
    print(render())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
