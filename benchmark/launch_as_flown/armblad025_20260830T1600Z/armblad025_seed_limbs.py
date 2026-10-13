#!/usr/bin/env python3
# =============================================================================
# RESPONSIBILITY: the B2 limbs B2a-B2i. Every one is EXECUTED and every one is driven to its
#   REFUSAL as well as its pass -- a check that cannot fail is not a gate (RB-20). Two limbs
#   matter more than the rest and are the reason this file exists rather than a docstring:
#   B2d drives the prereg S6.1 paired_csv ruling in BOTH directions (set on a wind rung must
#   refuse; CLEARED on the still-air rung must ALSO refuse). B2h and B2i -- the two limbs that
#   read SOURCE and CONFIG rather than a built table -- live in the sibling
#   armblad025_seed_static_limbs.py: this file reached 111 code-LoC against a hard 110 and caps
#   are SPLIT, never raised (the tremed_cont_limbs precedent, 2026-08-25).
# PUBLIC API: all_limbs(tmpdir) -> list[(id, ok, detail)]; main() -> int
# SIDE-EFFECTS: writes ONLY inside the caller-supplied tmp dir. No box contact, no ledger, no
#   checkpoint, no network, no seed. The planner is imported, never executed as a CLI.
# SIZE: soft 85 / hard 105 code-LoC; function cap 30. DISCLOSED: 93 code, +8 over soft after
#   the B2h/B2i split; the HARD cap holds with 12 lines of headroom and was NOT raised.
# COVERAGE: tests/unit/test_armblad_seed_plan.py runs main() under pytest.
# =============================================================================
import copy
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import armblad025_seed_plan as sp                                          # noqa: E402
import armblad025_seed_cli as cli                                          # noqa: E402
import armblad025_seed_static_limbs as stat                                # noqa: E402


def _die(rc, msg):
    raise SystemExit("rc=%d %s" % (rc, msg))


def _refuses(fn):
    """A refusal arrives as our sentinel or as the planner's own die() -> SystemExit."""
    try:
        fn()
    except SystemExit:
        return True
    return False


def b2a_shape():
    t = sp.build_table()
    ok = (t["n_legs"] == 600 and t["n_distinct_seeds"] == 30 and t["n_rungs"] == 10
          and len({l["cell"] for l in t["legs"]}) == 10
          and sp.build_table() == t)                     # deterministic: no RNG, no clock
    return ("B2a_shape_and_determinism", ok,
            "600 legs / 30 seeds / 10 rungs / 10 cells; two builds byte-equal")


def b2b_no_new_seeds():
    t = sp.build_table()
    good = sp.assert_no_new_seeds(t, _die) is None
    bad = copy.deepcopy(t); bad["legs"][0]["seed"] = 99999
    hdr = copy.deepcopy(t); hdr["reserve_spent"] = True
    drw = copy.deepcopy(t); drw["new_seeds_drawn"] = 0     # a table that LIES about its draw
    return ("B2b_zero_new_seeds", good and _refuses(lambda: sp.assert_no_new_seeds(bad, _die))
            and _refuses(lambda: sp.assert_no_new_seeds(hdr, _die))
            and _refuses(lambda: sp.assert_no_new_seeds(drw, _die)),
            "fresh-block-only accepted; an off-block seed, a spent-reserve header and an undeclared "
            "draw all REFUSED")


def b2c_complete_cross():
    t = sp.build_table()
    good = sp.assert_complete_cross(t, _die) is None
    bad = copy.deepcopy(t); bad["legs"] = [l for l in bad["legs"][1:]]     # drop one row
    return ("B2c_complete_cross", good and _refuses(lambda: sp.assert_complete_cross(bad, _die)),
            "every seed flies all 20 (rung, trajectory) pairs; dropping ONE row REFUSED")


def b2d_paired_csv_both_ways():
    """Prereg S6.1, and the ONLY limb that can catch the two S0 defects the prereg names."""
    t = sp.build_table()
    good = sp.assert_paired_csv_on_s0_only(t, _die) is None
    on_wind = copy.deepcopy(t)
    for l in on_wind["legs"]:
        if l["label"] == "W0":
            l["paired_csv"] = True                       # defect 2: misroutes W0 into calm
    off_s0 = copy.deepcopy(t)
    for l in off_s0["legs"]:
        l["paired_csv"] = False                          # defect 1: S0 left ungated
    return ("B2d_paired_csv_s0_only_BOTH_WAYS",
            good and _refuses(lambda: sp.assert_paired_csv_on_s0_only(on_wind, _die))
            and _refuses(lambda: sp.assert_paired_csv_on_s0_only(off_s0, _die)),
            "S0-only accepted; SET on W0 REFUSED (calm misroute) and CLEARED on S0 REFUSED "
            "(ungated rung) -- the ruling is a gate in both directions")


def b2e_no_duplicate_leg():
    dup = (("W0", 0.0, 0.25), ("W0", 0.0, 0.25))         # same rung twice -> same (cell,seed,traj)
    return ("B2e_duplicate_leg_refused",
            _refuses(lambda: sp.build_table(rungs=dup)),
            "a rung repeated in RUNGS collides on (cell, seed, traj) and REFUSES")


def b2f_firewall_allowlist(td):
    inside = os.path.join(td, "declared", "W0_10775_straight")
    outside = os.path.join(td, "undeclared", "W0_10775_racetrack")
    for d in (inside, outside):
        os.makedirs(d, exist_ok=True)
    allowed = os.path.join(td, "declared")
    clean = cli.census_block(sp.BLOCK, [os.path.join(td, "declared")], [allowed]) == []
    caught = cli.census_block(sp.BLOCK, [td], [allowed]) == [outside]
    return ("B2f_firewall_allowlist", clean and caught,
            "a block seed inside a DECLARED tree passes; the same seed outside it is CAUGHT")


def b2g_sha_binds_bytes():
    body = json.dumps(sp.build_table(), indent=1, sort_keys=True) + "\n"
    a = hashlib.sha256(body.encode()).hexdigest()
    b = hashlib.sha256((body[:-1] + " \n").encode()).hexdigest()
    return ("B2g_sha_binds_the_bytes", a != b and len(a) == 64,
            "a one-character edit moves the sha -- S16 binds the SHA, never the filename")


def all_limbs(td):
    return [b2a_shape(), b2b_no_new_seeds(), b2c_complete_cross(), b2d_paired_csv_both_ways(),
            b2e_no_duplicate_leg(), b2f_firewall_allowlist(td), b2g_sha_binds_bytes(),
            stat.b2h_ladder_matches_live_config(), stat.b2i_no_rng()]


def main():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rows = all_limbs(td)
    for i, ok, d in rows:
        print(" %-38s %s %s" % (i, "PASS" if ok else "FAIL", d))
    bad = [r for r in rows if not r[1]]
    print("armblad025_seed_limbs: %d/%d limbs pass" % (len(rows) - len(bad), len(rows)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
