#!/usr/bin/env python3
"""armblad025_selfcheck -- every ARMBLAD-025 authority gate, driven BOTH directions.

RESPONSIBILITY
  A gate that cannot refuse is not a gate (RB-20). This file drives each one to its refusal as
  well as its pass, on fixtures. The four SIGNATURE limbs S1-S4 live in the sibling
  armblad025_sig_limbs.py: this file reached 133 code-lines against a hard 125 and caps are
  SPLIT, never raised. S3 -- an all-INDENTED document must satisfy NOTHING -- is the single most
  valuable limb in the pair and is the reason the split kept them together rather than scattering.

PUBLIC API: all_limbs(tmpdir) -> list[(id, ok, detail)]; main() -> int
SIDE-EFFECTS: writes ONLY inside the caller-supplied tmp dir. Reads the real prereg and table
  READ-ONLY. Spawns nothing, kills nothing, contacts no box, flies nothing.
CAPS: soft 100 / hard 125 code-lines; function cap 30.
COVERAGE: run directly AFTER the operator countersigns (S1 reads the REAL signed prereg; S2-S4 fixture unsigned/tampered copies). Deliberately NOT collected by pytest: a limb that needs the operator's signature must not sit in the suite RED until it is given.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import armblad025_calm_witness as cw                                       # noqa: E402
import armblad025_manifest as bm                                           # noqa: E402
import armblad025_sig_limbs as sl                                          # noqa: E402
import armblad025_venue_limb as vl                                          # noqa: E402

REAL_DOC = os.path.join(HERE, "ARMBLAD025_PREREG.md")
REAL_TAB = os.path.join(HERE, "SEED_CELL_TABLE_ARMBLAD025.json")
REAL_MAN = os.path.join(HERE, "SLOT_MANIFEST_ARMBLAD025.json")
DRIVER = os.path.join(HERE, "armblad025_driver.sh")


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


def s5_manifest_gate_both_ways(td):
    rc0, _ = bm.gate(REAL_MAN, REAL_TAB, REAL_DOC)
    hacked = os.path.join(td, "hacked_manifest.json")
    m = json.load(open(REAL_MAN))
    m["legs"][0]["env_id"] = 1 + (m["legs"][0]["env_id"] % 24)
    json.dump(m, open(hacked, "w"))
    rc1, why1 = bm.gate(hacked, REAL_TAB, REAL_DOC)
    absent, _ = bm.gate(os.path.join(td, "nope.json"), REAL_TAB, REAL_DOC)
    return ("S5_manifest_gate_both_ways", rc0 == 0 and rc1 == 1 and absent == 2,
            "real manifest EQUALS its recomputation (rc 0); one re-dealt leg REFUSES (rc 1); "
            "an absent manifest is rc 2")


def s6_paired_csv_survives_to_the_driver():
    m = json.load(open(REAL_MAN))
    s0 = [l for l in m["legs"] if l["paired_csv"]]
    labels = {l["label"] for l in s0}
    others = {l["paired_csv"] for l in m["legs"] if l["label"] != "S0"}
    return ("S6_paired_csv_reaches_the_manifest",
            len(s0) == 60 and labels == {"S0"} and others == {False},
            "60 S0 rows carry paired_csv; every other rung carries False -- the S6.1 ruling "
            "survives table -> manifest -> driver without being re-derived")


def s7_calm_witness_catches_the_silent_fallback(td):
    cases, got = {}, {}
    for name, body in (("never", None),
                       ("tier1", {"rc": 0, "sigma_ned": [0, 0, 0]}),
                       ("confirmed", {"calm_commanded": True, "state": "CALM_CONFIRMED"}),
                       ("delivered", {"calm_commanded": True, "state": "ASSERT_FAIL"})):
        d = os.path.join(td, "cw_" + name)
        os.makedirs(d, exist_ok=True)
        if body is not None:
            json.dump(body, open(os.path.join(d, "sigma_gate.json"), "w"))
        cases[name] = d
        got[name] = cw.assert_calm_engaged(d)[0]
    return ("S7_calm_witness_both_ways",
            got == {"never": 5, "tier1": 5, "confirmed": 0, "delivered": 3},
            "a MISSING gate file and a tier-1 verdict both read rc 5 (UNGATED); "
            "CALM_CONFIRMED 0; a calm cell that delivered wind 3")


def s8_driver_dryrun_plans_its_slot():
    r = subprocess.run(["bash", DRIVER, "dryrun", "7", "24"], capture_output=True, text=True,
                       cwd=FD)
    lines = [l for l in r.stdout.splitlines() if l.startswith("plan[e7:")]
    return ("S8_driver_dryrun_plans_25_legs", r.returncode == 0 and len(lines) == 25,
            "slot 7 plans %d legs, spawns no SITL, rc %d" % (len(lines), r.returncode))


def all_limbs(td):
    return [sl.s1_real_doc_passes(), sl.s2_missing_signatures_refuse(td),
            sl.s3_indented_templates_satisfy_nothing(td), sl.s4_sha_drift_refuses(td),
            s5_manifest_gate_both_ways(td), s6_paired_csv_survives_to_the_driver(),
            s7_calm_witness_catches_the_silent_fallback(td), s8_driver_dryrun_plans_its_slot(),
            vl.s9_venue_satisfies_eval_contract()]


def main():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rows = all_limbs(td)
    for i, ok, d in rows:
        print(" %-42s %s %s" % (i, "PASS" if ok else "FAIL", d))
    bad = [r for r in rows if not r[1]]
    print("armblad025_selfcheck: %d/%d limbs pass" % (len(rows) - len(bad), len(rows)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
