#!/usr/bin/env python3
# PIDO TWIN of analysis/cs600_20260903T0230Z/cs600_manifest_check.py (parent NOT edited) with ONE declared delta: the only
#   signable knob shape is the PAIR {"voff": 0.25, "scales": 0.0} (both required, nothing else), and the gate prints BOTH
#   flags the fire script must forward: "--vert-offset-down-m 0.25 --action-scales 0.00". A null knob, a single knob, or
#   any other value refuses 36: this dispatch flies the PID-baseline + offset and nothing else.
# RESPONSIBILITY: refuse a PIDO run whose (policy sha, table sha, dose, knob pair) is not the SIGNED one.
# PUBLIC API: --manifest <path> --run <PIDO25|PIDO35> --ckpt <path> --table <path> --dose <float>
#   rc 0 signed quadruple (stdout last line: KNOBFLAG=<both flags>). 31 unknown run. 32 wrong policy. 33 wrong table.
#   34 wrong dose. 35 manifest absent/unreadable. 36 knob shape != {"voff": 0.25, "scales": 0.0}
# SIDE-EFFECTS: none (reads the manifest and hashes two files).
# SIZE: soft 45 / hard 60 code-lines; function cap 30.
# COVERAGE: pido_selfcheck.py -- both signed rows print both flags; sibling table 33; wrong dose 34; unknown run 31;
#   absent manifest 35; knob {voff 0.25} alone, {scales 0.0} alone, {voff 0.25, scales 1.0} and null all 36.
import argparse, hashlib, json, sys
REQUIRED = (("voff", "--vert-offset-down-m", 0.25), ("scales", "--action-scales", 0.0))

def sha256_file(p):
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()

def knob_flag(knob):
    if not isinstance(knob, dict) or sorted(knob) != sorted(k for k, _, _ in REQUIRED): return None
    for k, _, v in REQUIRED:
        if abs(float(knob[k]) - v) >= 1e-9: return None
    return " ".join("%s %.2f" % (f, v) for _, f, v in REQUIRED)

def check(manifest, run, ckpt, table, dose):
    try: rows = json.load(open(manifest))["runs"]
    except (OSError, KeyError, ValueError) as e: return 35, "manifest absent/unreadable: %s (%s)" % (manifest, type(e).__name__), ""
    r = rows.get(run)
    if r is None: return 31, "run %r is not in the signed manifest (%s)" % (run, sorted(rows)), ""
    flag = knob_flag(r.get("knob"))
    if flag is None: return 36, "run %s carries knob %r -- only {'voff': 0.25, 'scales': 0.0} (both) is signable here" % (run, r.get("knob")), ""
    if sha256_file(ckpt) != r["ckpt_sha"]: return 32, "run %s is signed for policy %s.., not this checkpoint" % (run, r["ckpt_sha"][:16]), ""
    if sha256_file(table) != r["table_sha"]: return 33, "run %s is signed for table %s.. (%s), not this table" % (run, r["table_sha"][:16], r["table_label"]), ""
    if abs(float(dose) - float(r["dose"])) > 1e-9: return 34, "run %s is signed at dose %.2f, not %.2f" % (run, float(r["dose"]), float(dose)), ""
    return 0, "MANIFEST OK: run %s = %s (%s..) on %s, dose %.2f, knob %s" % (run, r["policy"][:60], r["ckpt_sha"][:16], r["table_label"], float(r["dose"]), r.get("knob")), flag

def main(argv=None):
    ap = argparse.ArgumentParser(description="signed (policy, table, dose, knob pair) gate")
    for f in ("--manifest", "--run", "--ckpt", "--table", "--dose"): ap.add_argument(f, required=True)
    a = ap.parse_args(argv)
    rc, msg, flag = check(a.manifest, a.run, a.ckpt, a.table, a.dose)
    print(msg); print("KNOBFLAG=%s" % flag)
    return rc

if __name__ == "__main__":
    sys.exit(main())
