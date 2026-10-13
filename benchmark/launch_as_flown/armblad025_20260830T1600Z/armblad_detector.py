#!/usr/bin/env python3
"""armblad_detector -- the L1 failure modes of the two FRESH ARM-B ladders (ARMBLAD-025 / -035), each
driven to a PASS and a REFUSAL. A check that cannot fail is not a gate (RB-20).

  F1 an EXISTING signature authorises a fresh table: a BASELINE025-SEEDS-<fresh sha> clause must be
      REFUSED by the fresh gate, and an ARMBLAD025-SEEDS-<parent sha> clause REFUSED by the parent's.
  F2 table drift after signing: one appended byte REFUSES; the untouched table passes (fixture doc).
  F3 a fresh label has already flown: the on-disk census of dev analysis/ by NUMERIC range reads 0 hits
      outside the two ARMBLAD roots; a planted junk-named dir IS caught.
  F4 the derivation changed behaviour: the derived planner handed the PARENT block reproduces the
      parent's countersigned table LEG FOR LEG (the header differs ONLY in the four declared fields).
  F5 the control rung mis-gated: paired_csv on S0 only (both ways) and the calm witness both ways.
  F6 the sigma-0.00 control widens T5: ALLOWED_SIGMAS is still (0.25, 0.35) and 0.20/0.30/0.40 rows
      are still refused by the lineage's sigma predicate.
  F7 the two control rungs are not replicas: the S0 rows deal to IDENTICAL slots in both tables.
  F8 the lane-log clash recurs: both derived drivers carry the trajectory-keyed form, never the old.
PUBLIC API: main() -> 0 all PASS, 1 any FAIL. SIDE-EFFECTS: reads only; writes only under a tmp dir.
CAPS: file soft 95 / hard 115 code-lines; function cap 40. COVERAGE: this file IS the detector.
"""
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile

FD = "/home/ubuntu/FlightDock"
L25 = os.path.join(FD, "analysis/armblad025_20260830T1600Z")
L35 = os.path.join(FD, "analysis/armblad035_20260830T1600Z")
P25 = os.path.join(FD, "analysis/baseline025_20260826T1900Z")
P35 = os.path.join(FD, "analysis/baseline035_20260827T0900Z")
sys.path.insert(0, os.path.join(FD, "analysis/armbsig_20260824T0030Z"))
import armbsig_grids as g                                                   # noqa: E402
from armbsig_manifest import assign                                         # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def refuses(fn):
    try:
        fn(); return False
    except SystemExit:
        return True


def f1_f2(td, lad, par, tag, ptag):
    sg, psg = load(os.path.join(lad, tag.lower() + "_sig_gates.py"), tag + "sg"), \
              load(os.path.join(par, ptag.lower() + "_sig_gates.py"), ptag + "sg")
    fresh, parent = sg.file_sha(sg.TABLE), psg.file_sha(psg.TABLE)
    doc_wrong = os.path.join(td, "wrong_%s.md" % tag)
    open(doc_wrong, "w").write("OPERATOR-SIGNATURE: %s-SEEDS-x - %s; clause\n" % (ptag, fresh))
    doc_cross = os.path.join(td, "cross_%s.md" % tag)
    open(doc_cross, "w").write("OPERATOR-SIGNATURE: %s-SEEDS-x - %s; clause\n" % (tag, parent))
    doc_ok = os.path.join(td, "ok_%s.md" % tag)
    open(doc_ok, "w").write("OPERATOR-SIGNATURE: %s-SEEDS-x - %s; clause\n" % (tag, fresh))
    f1 = (refuses(lambda: sg.assert_countersigned(sg.TABLE, doc_wrong))
          and refuses(lambda: psg.assert_countersigned(psg.TABLE, doc_cross)))
    drifted = os.path.join(td, "drift_%s.json" % tag); shutil.copy2(sg.TABLE, drifted)
    open(drifted, "a").write(" \n")
    f2 = sg.assert_countersigned(sg.TABLE, doc_ok) == fresh and refuses(lambda: sg.assert_countersigned(drifted, doc_ok))
    return [("F1 %s: parent-prefixed clause REFUSED; fresh clause on the parent REFUSED" % tag, f1),
            ("F2 %s: fresh clause passes on the real bytes; one appended byte REFUSES" % tag, f2)]


def f3(td):
    cli = load(os.path.join(L25, "armblad025_seed_cli.py"), "cli25")
    clean = cli.census_block(range(10775, 10805), [os.path.join(FD, "analysis")], [L25, L35]) == []
    junk = os.path.join(td, "junk", "B25-W0_10780_straight"); os.makedirs(junk)
    return [("F3 census dev analysis/ 0 hits outside the two ARMBLAD roots; a planted 10780 dir IS caught",
             clean and cli.census_block(range(10775, 10805), [td], [L25, L35]) == [junk])]


def f4(lad, par, tag, ptag):
    sp, psp = load(os.path.join(lad, tag.lower() + "_seed_plan.py"), tag), load(os.path.join(par, ptag.lower() + "_seed_plan.py"), ptag)
    d, p = sp.build_table(block=psp.BLOCK), json.load(open(os.path.join(par, "SEED_CELL_TABLE_%s.json" % ptag)))
    strip = lambda t: [{k: v for k, v in l.items() if k != "origin"} for l in t["legs"]]
    hdr = lambda t: {k: v for k, v in t.items() if k not in ("legs", "dispatch", "new_seeds_drawn", "block")}
    return [("F4 %s: derived planner on the PARENT block == parent table leg for leg; header equal but the declared fields" % tag,
             strip(d) == strip(p) and hdr(d) == hdr(p) and d["block"] == p["block"] and d["n_legs"] == 600)]


def f5_f7(td):
    sl25 = load(os.path.join(L25, "armblad025_seed_limbs.py"), "sl25"); sc25 = load(os.path.join(L25, "armblad025_selfcheck.py"), "sc25")
    ok5 = sl25.b2d_paired_csv_both_ways()[1] and sc25.s7_calm_witness_catches_the_silent_fallback(td)[1]
    t25 = json.load(open(os.path.join(L25, "SEED_CELL_TABLE_ARMBLAD025.json")))["legs"]
    t35 = json.load(open(os.path.join(L35, "SEED_CELL_TABLE_ARMBLAD035.json")))["legs"]
    s0 = lambda legs: sorted((l["seed"], l["traj"], l["env_id"]) for l in assign(legs, 24) if l["label"] == "S0")
    a, b = s0(t25), s0(t35)
    return [("F5 paired_csv S0-only BOTH ways; calm witness: missing/tier-1 -> 5, CONFIRMED -> 0, delivered wind -> 3", ok5),
            ("F7 the 60 S0 legs deal to IDENTICAL slots in both ladders (replica control)", a == b and len(a) == 60)]


def f6():
    rows = json.load(open(os.path.join(L25, "SEED_CELL_TABLE_ARMBLAD025.json")))["legs"]
    bad = [dict(rows[0], sigma_ms=s) for s in (0.20, 0.30, 0.40)]
    return [("F6 ALLOWED_SIGMAS unchanged (0.25, 0.35); 0.20 / 0.30 / 0.40 rows still REFUSED by the lineage predicate",
             g.ALLOWED_SIGMAS == (0.25, 0.35) and all(g.sigma_refusal([b], "table") is not None for b in bad))]


def f8():
    new, old = 'lane_${SEED}_${TRAJ}.log', 'lane_${SEED}.log'
    ok = all(open(p).read().count(new) >= 2 and old not in open(p).read()
             for p in (os.path.join(L25, "armblad025_driver.sh"), os.path.join(L35, "armblad035_driver.sh")))
    return [("F8 both derived drivers key the lane log on (seed, trajectory); the clashing form is absent", ok)]


def main():
    td = tempfile.mkdtemp(prefix="armblad_det_")
    try:
        res = (f1_f2(td, L25, P25, "ARMBLAD025", "BASELINE025") + f1_f2(td, L35, P35, "ARMBLAD035", "BASELINE035")
               + f3(td) + f4(L25, P25, "ARMBLAD025", "BASELINE025") + f4(L35, P35, "ARMBLAD035", "BASELINE035")
               + f5_f7(td) + f6() + f8())
    finally:
        shutil.rmtree(td, ignore_errors=True)
    for n, ok in res:
        print("%s %s" % ("PASS" if ok else "FAIL", n))
    return 0 if all(ok for _, ok in res) else 1


if __name__ == "__main__":
    sys.exit(main())
