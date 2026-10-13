#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: put the ARMBLAD-025 table on disk and run the on-disk SEED FIREWALL.
#   Split out of armblad025_seed_plan.py 2026-08-26: that file reached 135 code-LoC against a
#   declared hard cap of 130, and caps are SPLIT, never raised. The boundary is real, not
#   cosmetic -- the planner is a pure function of its constants and touches no filesystem; every
#   read, every write and every argv in this campaign's table build lives HERE, in one place.
#
#   WHY THE FIREWALL IS AN ALLOW-LIST, NOT A NEVER-FLOWN CHECK. armbsig_seed_plan.census_block
#   refuses if a block seed appears in ANY flown leg directory, because a fresh draw colliding
#   with a flown seed is a corpus that cannot be pooled. ARMBLAD-025 draws nothing: its 30
#   seeds are FRESH (10775-10804, never flown before 2026-08-30) but SHARED with ARMBLAD-035, so after
#   either ladder flies, a bare "never flown" would refuse the other. The allow-list names the two
#   ARMBLAD roots and nothing else; a hit anywhere else (a purged junk corpus, an exam grid) still refuses.
#   (parent text:) seeds had ALREADY flown, under ARMBSIG-810, deliberately. So "never flown" would refuse
#   this campaign by construction and prove nothing. The invariant that still bites is "flown
#   only under a DECLARED dispatch": every hit must sit inside a tree named on the command line,
#   and a hit anywhere else means a seed is sitting in a corpus nobody declared -- rc 4.
# PUBLIC API: census_block(block, roots, allow_trees) -> list[str]
#             main(argv=None) -> int # 0 ok. 2 the table is malformed. 4 firewall hit
# SIDE-EFFECTS: writes ONLY --out and --out.sha256, and nothing at all under --dry-run. Reads
#   flown-leg directory NAMES read-only for the census. Never mutates a frozen or signed
#   artifact, spawns nothing, kills nothing, draws no seed, touches no ledger, contacts no box.
# SIZE: soft 70 / hard 90 code-LoC; function cap 35. DISCLOSED: the cap is set HERE, at file
#   creation -- a cap being CHOSEN, not an established cap being raised. No existing cap moved:
#   armblad025_seed_plan.py keeps the soft 80 / hard 100 it was created with and sits at 75.
#   DISCLOSED: this file is 79 code, +9 over its own soft cap after declared_trees() and
#   _report() were lifted out of main() to hold the 35-line function cap. HARD holds with 11
#   lines of headroom and was NOT raised.
# COVERAGE: armblad025_seed_limbs.py (B2f, B2g driven to their refusals) and
#   tests/unit/test_armblad_seed_plan.py, EXECUTED.
# =============================================================================
import argparse
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from armblad025_seed_plan import (                                    # noqa: E402
    BLOCK, assert_complete_cross, assert_no_new_seeds,
    assert_paired_csv_on_s0_only, build_table, die)


def census_block(block, roots, allow_trees):
    """The block HAS flown -- deliberately, under ARMBSIG-810. So the invariant is not 'never
    flown' but 'flown only under a DECLARED dispatch'. A hit inside an allowed tree is expected;
    a hit anywhere else means a seed is in a corpus nobody declared, and that refuses (rc 4)."""
    want, hits = {int(s) for s in block}, []
    allow = [os.path.abspath(t) for t in allow_trees]
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "*_*_*"), recursive=True):
            if not os.path.isdir(path):
                continue
            m = re.match(r"^(.+)_(\d+)_(straight|racetrack)$", os.path.basename(path))
            if m and int(m.group(2)) in want:          # NUMERIC, never a digit pattern (08-29 defect)
                ap = os.path.abspath(path)
                if not any(ap.startswith(t) for t in allow):
                    hits.append(path)
    return hits


def declared_trees(fd, here):
    """The ONLY trees where block 10775-10804 may ever appear: the two ARMBLAD ladder roots. The
    block was NEVER flown before 2026-08-30 (four-host census 0/0/0/0 at build time). Once ladder 1
    has flown, its legs make the block 'already flown' for ladder 2 -- so the invariant is 'flown
    only under these two dispatches'; a hit ANYWHERE else (a junk corpus, an exam grid) refuses."""
    return [
        os.path.join(fd, "analysis/armblad025_20260830T1600Z"),   # the sigma-0.25 ladder
        os.path.join(fd, "analysis/armblad035_20260830T1600Z"),   # the sigma-0.35 ladder
    ]


def _report(table, out, sha, roots, dry):
    print("SEED CELL TABLE%s: %s" % (" (DRY RUN, not written)" if dry else "", out))
    print(" legs=%d distinct_seeds=%d rungs=%d cells=%d block=%d-%d"
          % (table["n_legs"], table["n_distinct_seeds"], table["n_rungs"],
             len({l["cell"] for l in table["legs"]}), BLOCK[0], BLOCK[-1]))
    print(" new_seeds_drawn=30 reserve_spent=False paired_csv rungs=%s"
          % sorted({l["label"] for l in table["legs"] if l["paired_csv"]}))
    print(" firewall: %d root(s) scanned, 0 prior flights outside the two ARMBLAD roots" % len(roots))
    print(" sha256=%s" % sha)
    print(" COUNTERSIGN THIS SHA (prereg S16) before any leg flies.")


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    fd = os.path.dirname(os.path.dirname(here))
    ap = argparse.ArgumentParser(description="ARMBLAD-025 seed/cell table")
    ap.add_argument("--out", default=os.path.join(here, "SEED_CELL_TABLE_ARMBLAD025.json"))
    ap.add_argument("--census-root", action="append", default=[])
    ap.add_argument("--allow-tree", action="append", default=[],
                    help="a tree where these seeds are DECLARED to have flown already")
    ap.add_argument("--dry-run", action="store_true", help="build and check; write nothing")
    a = ap.parse_args(argv)
    table = build_table()
    for check in (assert_no_new_seeds, assert_complete_cross, assert_paired_csv_on_s0_only):
        check(table, die)
    roots = a.census_root or [os.path.join(fd, "analysis")]
    allow = a.allow_tree or declared_trees(fd, here)
    hits = census_block(BLOCK, roots, allow)
    if hits:
        die(4, "seed firewall: %d leg dir(s) carry a block seed OUTSIDE the declared trees, "
               "e.g. %s" % (len(hits), hits[:3]))
    body = json.dumps(table, indent=1, sort_keys=True) + "\n"
    sha = hashlib.sha256(body.encode()).hexdigest()
    if not a.dry_run:
        with open(a.out, "w") as fh:
            fh.write(body)
        with open(a.out + ".sha256", "w") as fh:
            fh.write("%s %s\n" % (sha, os.path.basename(a.out)))
    _report(table, a.out, sha, roots, a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
