#!/usr/bin/env python3
"""armblad025_sig_limbs -- the four SIGNATURE limbs S1-S4, each driven BOTH directions.

RESPONSIBILITY
  Split out of armblad025_selfcheck.py 2026-08-26 at 133 code-lines against a hard 125 -- caps
  are SPLIT, never raised. These four belong together: they all answer "what did the operator
  actually sign, and can a near-miss slip past?"

  S3 IS THE LOAD-BEARING ONE. The prereg deliberately carries INDENTED templates of the very
  lines the gates grep for, so an author who writes the pattern without '^' gets a regex that
  matches a TEMPLATE and green-lights an UNSIGNED campaign. S3 builds exactly that document --
  four perfectly-formed signature lines, all indented -- and asserts every gate still refuses.

PUBLIC API: s1_real_doc_passes(). s2_missing_signatures_refuse(td)
  . s3_indented_templates_satisfy_nothing(td). s4_sha_drift_refuses(td)
  Each returns (id, ok, detail).
SIDE-EFFECTS: writes ONLY inside the caller-supplied tmp dir; reads the real prereg and table
  READ-ONLY. Spawns nothing, kills nothing, flies nothing.
CAPS: soft 55 / hard 75 code-lines; function cap 30.
COVERAGE: registered by armblad025_selfcheck.all_limbs(), which the unit test executes.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import armblad025_sig_gates as sg                                          # noqa: E402

REAL_DOC = os.path.join(HERE, "ARMBLAD025_PREREG.md")
REAL_TAB = os.path.join(HERE, "SEED_CELL_TABLE_ARMBLAD025.json")


def _refuses(fn):
    try:
        fn()
    except SystemExit:
        return True
    return False


def _doc(td, name, body):
    p = os.path.join(td, name)
    open(p, "w").write(body)
    return p


def s1_real_doc_passes():
    """S17 is REPORTED, never required (the gate's own contract); the parent ANDed it in. See prereg S19."""
    ok = (sg.assert_signed_prereg(REAL_DOC) and sg.signed_width(REAL_DOC)["width"] == 24
          and sg.assert_countersigned(REAL_TAB, REAL_DOC))
    return ("S1_signed_prereg_accepted", bool(ok),
            "S15 signed, S18 width=24, S16 sha matches the table; S17 abort-warrant %s (OPTIONAL, "
            "unsigned => false/missed abort rates render NOT MEASURED)"
            % ("SIGNED" if sg.abort_warrant_signed(REAL_DOC) else "unsigned"))


def s2_missing_signatures_refuse(td):
    empty = _doc(td, "empty.md", "# no signatures here\n")
    return ("S2_unsigned_doc_refused",
            all([_refuses(lambda: sg.assert_signed_prereg(empty)),
                 _refuses(lambda: sg.signed_width(empty)),
                 _refuses(lambda: sg.assert_countersigned(REAL_TAB, empty)),
                 sg.abort_warrant_signed(empty) is False]),
            "S15/S16/S18 all REFUSE on an unsigned doc; S17 reports False without refusing")


def s3_indented_templates_satisfy_nothing(td):
    """THE limb. The prereg carries these lines INDENTED on purpose."""
    body = ("# templates only\n"
            " OPERATOR-SIGNATURE: ARMBLAD025-20260826 - I sign this PREREG\n"
            " OPERATOR-SIGNATURE: ARMBLAD025-SEEDS-x - %s; clause\n"
            " OPERATOR-SIGNATURE: ARMBLAD025-WIDTH-<24> - I authorize\n"
            " OPERATOR-SIGNATURE: ARMBLAD025-ABORT-WARRANT-x - I sign\n" % ("f" * 64))
    tpl = _doc(td, "templates.md", body)
    return ("S3_indented_templates_satisfy_NOTHING",
            all([_refuses(lambda: sg.assert_signed_prereg(tpl)),
                 _refuses(lambda: sg.signed_width(tpl)),
                 _refuses(lambda: sg.assert_countersigned(REAL_TAB, tpl)),
                 sg.abort_warrant_signed(tpl) is False]),
            "an all-INDENTED doc satisfies no gate -- column-0 anchoring proven, not assumed")


def s4_sha_drift_refuses(td):
    drifted = os.path.join(td, "drifted_table.json")
    shutil.copy2(REAL_TAB, drifted)
    with open(drifted, "a") as fh:
        fh.write(" \n")
    good = sg.assert_countersigned(REAL_TAB, REAL_DOC)
    return ("S4_table_sha_drift_refused",
            bool(good) and _refuses(lambda: sg.assert_countersigned(drifted, REAL_DOC)),
            "the countersigned sha matches the real table; one appended byte REFUSES")


