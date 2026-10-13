#!/usr/bin/env python3
"""armblad035_sig_limbs -- the SIGNATURE limbs, each driven BOTH directions.

RESPONSIBILITY
  A gate that cannot refuse is not a gate (RB-20). These limbs answer one question: "what did
  the operator actually sign, and can a near-miss slip past?" They are split from the selfcheck
  for the same reason the parent split its own -- caps are SPLIT, never raised.

  S3 IS THE LOAD-BEARING ONE, inherited from the parent unchanged in spirit. The prereg
  deliberately carries INDENTED templates of the very lines the gates grep for, so an author
  who writes the pattern without '^' gets a regex that matches a TEMPLATE and green-lights an
  UNSIGNED campaign. S3 builds exactly that document -- perfectly-formed signature lines, all
  indented -- and asserts every gate still refuses.

  S10 -- the block-C band limb -- lives in the sibling armblad035_band_limb.py: this file
  reached 93 code-lines against a hard 90 and caps are SPLIT, never raised.
PUBLIC API: s1..s4 -- each returns (id, ok, detail)
SIDE-EFFECTS: writes ONLY inside the caller-supplied tmp dir; reads the real prereg and table
  READ-ONLY. Spawns nothing, kills nothing, flies nothing.
CAPS: soft 70 / hard 90 code-lines; function cap 30. DISCLOSED: 76 code, +6 over soft after
  the S10 split; the HARD cap holds with 14 lines of headroom and was NOT raised.
COVERAGE: registered by armblad035_selfcheck.all_limbs(), which the unit test executes.
NOT A TRUTH PATH: imports os/shutil/sys and this campaign's own gate module. Reaches no actor,
  critic, reward, replay buffer, observation or policy target.
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import armblad035_sig_gates as sg                                          # noqa: E402

REAL_DOC = os.path.join(HERE, "ARMBLAD035_PREREG.md")
REAL_TAB = os.path.join(HERE, "SEED_CELL_TABLE_ARMBLAD035.json")


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
    ok = (sg.assert_signed_prereg(REAL_DOC) and sg.assert_countersigned(REAL_TAB, REAL_DOC)
          and sg.assert_sigma_band_ruled(REAL_DOC)["mode"] in ("SCALED", "CLAMPED"))
    r = sg.assert_sigma_band_ruled(REAL_DOC)
    return ("S1_signed_prereg_accepted", bool(ok),
            "A signed, B sha matches the table, C ruled %s -> band %.4f, D %s"
            % (r["mode"], r["band"], "signed" if sg.s0_rung_signed(REAL_DOC) else "unsigned"))


def s2_missing_signatures_refuse(td):
    empty = _doc(td, "empty.md", "# no signatures here\n")
    return ("S2_unsigned_doc_refused",
            all([_refuses(lambda: sg.assert_signed_prereg(empty)),
                 _refuses(lambda: sg.assert_countersigned(REAL_TAB, empty)),
                 _refuses(lambda: sg.assert_sigma_band_ruled(empty)),
                 sg.s0_rung_signed(empty) is False]),
            "A/B/C all REFUSE on an unsigned doc; D reports False without refusing")


def s3_indented_templates_satisfy_nothing(td):
    """THE limb. The prereg carries these lines INDENTED on purpose."""
    body = ("# templates only\n"
            " OPERATOR-SIGNATURE: ARMBLAD035-20260827 - I sign this PREREG\n"
            " OPERATOR-SIGNATURE: ARMBLAD035-SEEDS-x - %s; clause\n"
            " OPERATOR-SIGNATURE: ARMBLAD035-SIGMABAND-20260827 - SCALED 0.1118\n"
            " OPERATOR-SIGNATURE: ARMBLAD035-S0-20260827 - I add S0\n" % ("f" * 64))
    tpl = _doc(td, "templates.md", body)
    return ("S3_indented_templates_satisfy_NOTHING",
            all([_refuses(lambda: sg.assert_signed_prereg(tpl)),
                 _refuses(lambda: sg.assert_countersigned(REAL_TAB, tpl)),
                 _refuses(lambda: sg.assert_sigma_band_ruled(tpl)),
                 sg.s0_rung_signed(tpl) is False]),
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
