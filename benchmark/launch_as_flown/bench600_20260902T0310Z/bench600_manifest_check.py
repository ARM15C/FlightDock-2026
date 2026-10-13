#!/usr/bin/env python3
# RESPONSIBILITY: refuse a BENCH600 run whose (policy sha, table sha, dose, knob=none) quadruple is
#   not the SIGNED one. Twin in spirit of knobsweep_exam_fire's knob_manifest_check.py (which
#   REQUIRES a knob and so cannot express election E1 "no exam knob"); this one binds the TABLE
#   instead, because the hazard here is different: two 600-leg tables (ARMBLAD025 sigma 0.25 and
#   ARMBLAD035 sigma 0.35) fly the SAME policy on two boxes, and a sigma-0.35 corpus mislabelled as
#   the H1/H2 run -- or the reverse -- would enter the wrong verdict. The table sha alone tells them
#   apart, so the gate refuses on it.
# PUBLIC API: --manifest <path> --run <H1H2|H3> --ckpt <path> --table <path> --dose <float>
#   rc 0 signed quadruple. 31 unknown run. 32 wrong policy. 33 wrong table. 34 wrong dose.
#   35 manifest absent/unreadable. 36 a knob field present in the row (E1 forbids one)
# SIDE-EFFECTS: none (reads the manifest and hashes two files).
# SIZE: soft 45 / hard 60 code-lines; function cap 25.
# COVERAGE: bench600_selfcheck.py -- both signed rows pass; sibling's table on the H1H2 row
#   refuses 33; wrong policy 32; wrong dose 34; unknown run 31; absent manifest 35; a row that
#   carries a knob refuses 36.
import argparse
import hashlib
import json
import sys


def sha256_file(p):
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def check(manifest, run, ckpt, table, dose):
    try:
        rows = json.load(open(manifest))["runs"]
    except (OSError, KeyError, ValueError) as e:
        return 35, "manifest absent/unreadable: %s (%s)" % (manifest, type(e).__name__)
    r = rows.get(run)
    if r is None:
        return 31, "run %r is not in the signed manifest (%s)" % (run, sorted(rows))
    if r.get("knob") not in (None, "none"):
        return 36, "run %s carries knob %r -- election E1 is 'no exam knob'" % (run, r["knob"])
    if sha256_file(ckpt) != r["ckpt_sha"]:
        return 32, "run %s is signed for policy %s.., not this checkpoint" % (run, r["ckpt_sha"][:16])
    if sha256_file(table) != r["table_sha"]:
        return 33, ("run %s is signed for table %s.. (%s), not this table -- REFUSING a sigma swap"
                    % (run, r["table_sha"][:16], r["table_label"]))
    if abs(float(dose) - float(r["dose"])) > 1e-9:
        return 34, "run %s is signed at dose %.2f, not %.2f" % (run, float(r["dose"]), float(dose))
    return 0, ("MANIFEST OK: run %s = %s (%s..) on %s, dose %.2f, no knob"
               % (run, r["policy"], r["ckpt_sha"][:16], r["table_label"], float(r["dose"])))


def main(argv=None):
    ap = argparse.ArgumentParser(description="signed (policy, table, dose, no-knob) gate")
    for f in ("--manifest", "--run", "--ckpt", "--table", "--dose"):
        ap.add_argument(f, required=True)
    a = ap.parse_args(argv)
    rc, msg = check(a.manifest, a.run, a.ckpt, a.table, a.dose)
    print(msg)
    return rc


if __name__ == "__main__":
    sys.exit(main())
