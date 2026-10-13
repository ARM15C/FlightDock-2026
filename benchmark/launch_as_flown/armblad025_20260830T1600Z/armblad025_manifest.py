#!/usr/bin/env python3
"""armblad025_manifest -- build AND gate the ARMBLAD-025 slot manifest.

RESPONSIBILITY
  Assign every leg of the countersigned 600-row cell table to a slot (env_id 1..WIDTH) and
  refuse any manifest that is not EQUAL to a recomputation from signed inputs. Build and gate
  live in ONE file on purpose: they share the assignment rule, and a rule duplicated in two
  files is a rule that will drift.

  THE ASSIGNMENT RULE IS IMPORTED, NOT RE-IMPLEMENTED. armbsig_manifest.assign() deals
  canonically-ordered legs round-robin across slots and is already flown, selfchecked and
  frozen. Copying it here would be the second copy of a rule that must never diverge, so it is
  IMPORTED. That module is READ-ONLY to this campaign and is not modified.

  EVERY GATE INPUT IS SIGNED, AND EVERY SIGNATURE HAS A CONSUMER HERE:
  * S15 ARMBLAD025-<date> the prereg itself is signed -> assert_signed_prereg
  * S16 ARMBLAD025-SEEDS- the table's FILE-BYTES sha -> assert_countersigned
  * S18 ARMBLAD025-WIDTH-<N> the fan-out width -> signed_width
  * S17 ARMBLAD025-ABORT-WARRANT- OPTIONAL; presence is REPORTED, -> abort_warrant_signed
                                    never required, and never gates a leg
  A signature no gate consumes cannot refuse anything (the T-322 / ARMBTOP-GATE-674 lesson),
  which is why S17 is read here even though it gates nothing: the manifest RECORDS whether the
  campaign is authorised to compute false/missed abort rates, so the readout cannot invent it.

  The four gates themselves live in the sibling armblad025_sig_gates.py: this file reached 144
  code-lines against a declared hard 120 and caps are SPLIT, never raised. The boundary is real
  -- "what the operator signed" and "how legs are dealt to slots" are different concerns.

PUBLIC API
  build(table_path, width) -> dict. signed_width(doc) -> dict
  . assert_signed_prereg(doc) -> str. assert_countersigned(table_path, doc) -> str
  . abort_warrant_signed(doc) -> bool. gate(man_path, table_path, doc) -> (rc, why)
  . main(argv) -> int Exit 0 pass. 1 refused. 2 absent/IO. 3 usage.

SIDE-EFFECTS
  Reads signed documents and JSON READ-ONLY. Writes only --out on --build. Spawns nothing,
  signals nothing, kills nothing, draws no seed, contacts no box.

CAPS: file soft 95 / hard 120 code-lines; function cap 35. DISCLOSED: 102 code, +7 over soft
  after the sig-gate split; the HARD cap holds with 18 lines of headroom and was NOT raised.
COVERAGE: armblad025_manifest_limbs.py -- every gate driven BOTH directions.
"""
import argparse
import json
import os
import sys

FD = "/home/ubuntu/FlightDock"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(FD, "analysis/armbsig_20260824T0030Z"))
sys.path.insert(0, HERE)
from armbsig_manifest import assign                                        # noqa: E402
from armblad025_sig_gates import (                                        # noqa: E402
    MANIFEST, PREREG, TABLE, abort_warrant_signed, assert_countersigned,
    assert_signed_prereg, die, file_sha, signed_width)

DISPATCH = "ARMBLAD-025"

def build(table_path=TABLE, width=None, doc=PREREG):
    assert_signed_prereg(doc)
    sha = assert_countersigned(table_path, doc)
    width = signed_width(doc)["width"] if width is None else width
    t = json.load(open(table_path))
    legs = assign(t["legs"], width)
    depths = {}
    for lg in legs:
        depths[lg["env_id"]] = depths.get(lg["env_id"], 0) + 1
    if max(depths.values()) - min(depths.values()) > 1:
        die(1, "slot depths unbalanced: %s" % sorted(depths.items()))
    if any("paired_csv" not in lg for lg in legs):
        die(1, "a leg lost its paired_csv field in assignment -- the S6.1 ruling must survive "
               "the manifest or the driver has nothing to read")
    return {"dispatch": DISPATCH, "width": width, "n_legs": len(legs),
            "source_table": os.path.basename(table_path), "source_table_sha256": sha,
            "prereg_sha256": file_sha(doc), "abort_warrant_signed": abort_warrant_signed(doc),
            "slot_depths": {str(k): v for k, v in sorted(depths.items())}, "legs": legs}


def gate(man_path=MANIFEST, table_path=TABLE, doc=PREREG):
    """EQUALITY, not plausibility: the manifest must equal a recomputation from signed inputs."""
    if not os.path.isfile(man_path):
        return 2, "manifest absent at %s" % man_path
    got = json.load(open(man_path))
    want = build(table_path, got.get("width"), doc)
    if got != want:
        diffs = [k for k in set(got) | set(want) if got.get(k) != want.get(k)]
        return 1, "manifest != recomputation from signed inputs; fields differing: %s" % diffs
    if got["width"] != signed_width(doc)["width"]:
        return 1, "manifest width %s != signed width" % got["width"]
    return 0, ("manifest EQUALS recomputation: %d legs, width %d, table sha %s, abort-warrant %s"
               % (got["n_legs"], got["width"], got["source_table_sha256"][:16],
                  "SIGNED" if got["abort_warrant_signed"] else "unsigned (rates NOT MEASURED)"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="ARMBLAD-025 slot manifest")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--doc", default=PREREG)
    ap.add_argument("--out", default=MANIFEST)
    a = ap.parse_args(argv)
    if a.build:
        man = build(a.table, None, a.doc)
        with open(a.out, "w") as fh:
            fh.write(json.dumps(man, indent=1, sort_keys=True) + "\n")
        print("SLOT MANIFEST: %s\n legs=%d width=%d depths=%s abort_warrant=%s"
              % (a.out, man["n_legs"], man["width"],
                 sorted(set(man["slot_depths"].values())), man["abort_warrant_signed"]))
        return 0
    if a.gate:
        rc, why = gate(a.manifest, a.table, a.doc)
        print("MANIFEST GATE rc=%d: %s" % (rc, why))
        return rc
    ap.error("--build or --gate")


if __name__ == "__main__":
    sys.exit(main())
