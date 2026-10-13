#!/usr/bin/env python3
"""armblad035_manifest -- deal the 600 legs to slots, and GATE the dealing.

RESPONSIBILITY
  COPY, not import, of baseline025_manifest.py (L2: the parent flew as-is). Builds the slot
  manifest from SIGNED inputs only, and gates an on-disk manifest by EQUALITY against a
  recomputation -- not by plausibility. If the manifest differs from what the signed prereg and
  countersigned table produce, everything downstream refuses.
  Slot assignment is COMPOSED from the validated armbsig_manifest.assign, never re-implemented.

  WHAT DIFFERS FROM THE PARENT: width is not a signature in this campaign (the prereg carries
  no width block), so it is an argument defaulting to 24 and the manifest RECORDS what was
  used. The parent's abort-warrant field is replaced by the block-C sigma-band ruling and the
  block-D S0 election, both recorded so the readout cannot invent either authority for itself.
PUBLIC API: build(table_path, width, doc) -> dict; gate(man, table, doc) -> (rc, why)
SIDE-EFFECTS: --build writes ONLY --out. --gate writes nothing. Spawns nothing, flies nothing.
CAPS: file soft 80 / hard 100 code-lines; function cap 30.
COVERAGE: armblad035_selfcheck.py -- the gate driven BOTH directions on fixtures.
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
from armblad035_sig_gates import (                                        # noqa: E402
    MANIFEST, PREREG, TABLE, assert_countersigned, assert_signed_prereg,
    assert_sigma_band_ruled, die, file_sha, s0_rung_signed)

DISPATCH = "ARMBLAD-035"

def build(table_path=TABLE, width=None, doc=PREREG):
    assert_signed_prereg(doc)
    sha = assert_countersigned(table_path, doc)
    width = 24 if width is None else int(width)
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
            "prereg_sha256": file_sha(doc), "sigma_band": assert_sigma_band_ruled(doc)["mode"],
            "sigma_band_value": assert_sigma_band_ruled(doc)["band"],
            "s0_rung_signed": s0_rung_signed(doc),
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
    return 0, ("manifest EQUALS recomputation: %d legs, width %d, table sha %s, band %s "
               "%.4f, S0 %s" % (got["n_legs"], got["width"], got["source_table_sha256"][:16],
                                got["sigma_band"], got["sigma_band_value"],
                                "flown" if got["s0_rung_signed"] else "NOT flown"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="ARMBLAD-035 slot manifest")
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
        print("SLOT MANIFEST: %s\n legs=%d width=%d depths=%s band=%s %.4f S0=%s"
              % (a.out, man["n_legs"], man["width"],
                 sorted(set(man["slot_depths"].values())), man["sigma_band"],
                 man["sigma_band_value"], man["s0_rung_signed"]))
        return 0
    if a.gate:
        rc, why = gate(a.manifest, a.table, a.doc)
        print("MANIFEST GATE rc=%d: %s" % (rc, why))
        return rc
    ap.error("--build or --gate")


if __name__ == "__main__":
    sys.exit(main())
