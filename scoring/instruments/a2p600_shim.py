#!/usr/bin/env python3
# =============================================================================
# Dispatch: A2P600-20260903. RESPONSIBILITY: make one landed BENCH600 exam corpus READABLE by the
#   ladder readout of record (analysis/armbref_20260827T2100Z/armbref_readout.py) WITHOUT copying,
#   renaming or touching a single leg. The two trees name the same leg differently:
#     ARMBLAD ladder (what armbref_readout.walk builds):  <root>/<cell>/<label>_<seed>_<traj>
#                                                          e.g. legs/B25-W8/W8_10775_straight
#     BENCH600 exam wave (bench600_exam_table.lane_dir):  <root>/<cell.split("-")[0]>/<same leg>
#                                                          e.g. .../B25/W8_10775_straight
#   So this writes ONE symlink per cell -- shim/<cell> -> <corpus>/<prefix> -- and the readout's
#   own path join then resolves every leg. Ten links, no data movement, nothing writable.
#   WHY A SHIM AND NOT A FLAG: armbref_readout.py is a PINNED instrument (its own PINNED dict
#   digest-verifies five siblings and tests/unit/test_armbref_readout.py pins its behaviour);
#   editing it to accept a second layout would make the two 600-leg ladders and parent A2's two
#   600-leg runs cross-instrument comparisons. The shim keeps ONE instrument for all four.
#   TWO CONSUMERS, TWO OPPOSITE CONVENTIONS -- measured 2026-09-03, the banked "a reproduction limb
#   can pass by finding nothing" class: armbref_readout.walk joins <root>/<cell>/<leg>, but
#   windowed_rescore.rescore joins <root>/<leg> FLAT. Pointed at the nested shim, the windowed pass
#   looked in 600 paths that do not exist, returned win=None for every leg, and its own validation
#   limb reported "reproduced: True" because it only compares legs where BOTH values are present --
#   a vacuous pass on an empty set. So --shim builds the NESTED layout and --flat the FLAT one, and
#   each is asserted separately.
#   THE DENOMINATOR IS ASSERTED, NOT HOPED: build() refuses unless EVERY manifest leg resolves to a
#   real directory holding a results_eid*.json, and prints the count it proved. Zero legs raises.
#   The flat shim additionally asserts the file windowed_rescore actually reads
#   (env*/flight_state_ep*.csv.gz), so an unreadable telemetry channel refuses HERE, loudly, instead
#   of arriving downstream as a column of nulls.
# PUBLIC API: cells(manifest) -> dict ; build(corpus, manifest, shim) -> int legs_resolved ;
#   build_flat(corpus, manifest, flat) -> int legs_resolved
#   CLI: python a2p600_shim.py --corpus DIR --manifest JSON [--shim DIR] [--flat DIR]
#   rc 0 built . 2 usage . 3 a manifest leg does not resolve . 4 corpus/manifest absent
# SIDE-EFFECTS: creates <shim>/ and symlinks inside it. Reads the corpus read-only. No network,
#   no SITL, no ledger, no seed, no S3, nothing under src/ or logs/ or any protected/ dir.
# SIZE: soft 70 / hard 90 code-LoC; function cap 25.
# COVERAGE: a2p600_selfcheck.py -- both cell shapes (B25/B35) build 600; a corpus missing one
#   leg refuses 3 naming it; a shim over an empty dir refuses 3 rather than reporting 0 legs.
# =============================================================================
import argparse
import glob
import json
import os
import sys


def cells(manifest):
    """cell -> the BENCH600 prefix that holds its legs ('B25-W8' -> 'B25')."""
    return {leg["cell"]: leg["cell"].split("-")[0] for leg in manifest["legs"]}


def build(corpus, manifest, shim):
    os.makedirs(shim, exist_ok=True)
    for cell, prefix in cells(manifest).items():
        link, target = os.path.join(shim, cell), os.path.join(corpus, prefix)
        if os.path.islink(link):
            os.unlink(link)
        os.symlink(target, link)
    n = 0
    for leg in manifest["legs"]:
        d = os.path.join(shim, leg["cell"], "%s_%d_%s" % (leg["label"], leg["seed"], leg["traj"]))
        if not os.path.isdir(d):
            raise SystemExit("REFUSE rc 3: manifest leg does not resolve through the shim: %s" % d)
        if not glob.glob(os.path.join(d, "results_eid*.json")):
            raise SystemExit("REFUSE rc 3: no results_eid*.json under %s" % d)
        n += 1
    if n == 0:
        raise SystemExit("REFUSE rc 3: the shim resolved ZERO legs -- a denominator of 0 is not a pass")
    return n


def build_flat(corpus, manifest, flat):
    """windowed_rescore's convention: <root>/<leg>. Asserts the telemetry file it actually reads."""
    os.makedirs(flat, exist_ok=True)
    n = 0
    for leg in manifest["legs"]:
        name = "%s_%d_%s" % (leg["label"], leg["seed"], leg["traj"])
        target = os.path.join(corpus, leg["cell"].split("-")[0], name)
        link = os.path.join(flat, name)
        if os.path.islink(link):
            os.unlink(link)
        os.symlink(target, link)
        if not glob.glob(os.path.join(link, "env*", "flight_state_ep*.csv.gz")):
            raise SystemExit("REFUSE rc 3: no env*/flight_state_ep*.csv.gz under %s -- the windowed "
                             "channel would read as null for this leg" % link)
        n += 1
    if n == 0:
        raise SystemExit("REFUSE rc 3: the flat shim resolved ZERO legs")
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="symlink shims: BENCH600 corpus -> the two reader layouts")
    for f in ("--corpus", "--manifest"):
        ap.add_argument(f, required=True)
    ap.add_argument("--shim")
    ap.add_argument("--flat")
    a = ap.parse_args(argv)
    for p in (a.corpus, a.manifest):
        if not os.path.exists(p):
            print("REFUSE rc 4: absent: %s" % p)
            return 4
    if not (a.shim or a.flat):
        ap.error("one of --shim (nested) or --flat is required")
    m = json.load(open(a.manifest))
    if a.shim:
        print("NESTED SHIM OK: %d of %d manifest legs resolve through %s (10 symlinks, no data copied)"
              % (build(a.corpus, m, a.shim), len(m["legs"]), a.shim))
    if a.flat:
        print("FLAT SHIM OK: %d of %d manifest legs resolve through %s, each with its telemetry gz"
              % (build_flat(a.corpus, m, a.flat), len(m["legs"]), a.flat))
    return 0


if __name__ == "__main__":
    sys.exit(main())
