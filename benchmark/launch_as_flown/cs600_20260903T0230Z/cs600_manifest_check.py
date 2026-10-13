#!/usr/bin/env python3
# CS600 TWIN of analysis/bench600_20260902T0310Z/bench600_manifest_check.py with ONE declared delta: a run row MAY carry
#   knob = {"voff": 0.25} (the CANDSSW-leg1 configuration of record) and the gate then PRINTS the exact flag the fire
#   script must forward; a row with knob null prints NO flag. Any other knob shape refuses 36. The parent is NOT edited.
# RESPONSIBILITY: refuse a CS600 run whose (policy sha, table sha, dose, knob) quadruple is not the SIGNED one.
# PUBLIC API: --manifest <path> --run <CSP|CSV|CSC> --ckpt <path> --table <path> --dose <float>
#   rc 0 signed quadruple (stdout last line: KNOBFLAG=<flag or empty>). 31 unknown run. 32 wrong policy. 33 wrong table.
#   34 wrong dose. 35 manifest absent/unreadable. 36 knob shape not one of {null, {"voff": 0.25}, {"clamp": 0.50}}
# SIDE-EFFECTS: none (reads the manifest and hashes two files).
# SIZE: soft 45 / hard 60 code-lines; function cap 30.
# COVERAGE: cs600_selfcheck.py -- both signed rows pass (CSP prints an EMPTY flag, CSV prints --vert-offset-down-m 0.25);
#   sibling's policy on the other row 32; wrong table 33; wrong dose 34; unknown run 31; absent manifest 35; a knob value
#   OUTSIDE the signed set (e.g. clamp 0.25, voff 0.50, a bias row) 36.
import argparse, hashlib, json, sys
ALLOWED = {"voff": ("--vert-offset-down-m", 0.25), "clamp": ("--clamp-behind-positive-ms", 0.50)}

def sha256_file(p):
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()

def knob_flag(knob):
    if knob in (None, "none"): return ""
    if isinstance(knob, dict) and len(knob) == 1:
        (k, v), = knob.items()
        if k in ALLOWED and abs(float(v) - ALLOWED[k][1]) < 1e-9: return "%s %.2f" % (ALLOWED[k][0], ALLOWED[k][1])
    return None

def check(manifest, run, ckpt, table, dose):
    try: rows = json.load(open(manifest))["runs"]
    except (OSError, KeyError, ValueError) as e: return 35, "manifest absent/unreadable: %s (%s)" % (manifest, type(e).__name__), ""
    r = rows.get(run)
    if r is None: return 31, "run %r is not in the signed manifest (%s)" % (run, sorted(rows)), ""
    flag = knob_flag(r.get("knob"))
    if flag is None: return 36, "run %s carries knob %r -- only null, {'voff': 0.25} or {'clamp': 0.50} is signable here" % (run, r.get("knob")), ""
    if sha256_file(ckpt) != r["ckpt_sha"]: return 32, "run %s is signed for policy %s.., not this checkpoint" % (run, r["ckpt_sha"][:16]), ""
    if sha256_file(table) != r["table_sha"]: return 33, "run %s is signed for table %s.. (%s), not this table" % (run, r["table_sha"][:16], r["table_label"]), ""
    if abs(float(dose) - float(r["dose"])) > 1e-9: return 34, "run %s is signed at dose %.2f, not %.2f" % (run, float(r["dose"]), float(dose)), ""
    return 0, "MANIFEST OK: run %s = %s (%s..) on %s, dose %.2f, knob %s" % (run, r["policy"], r["ckpt_sha"][:16], r["table_label"], float(r["dose"]), r.get("knob")), flag

def main(argv=None):
    ap = argparse.ArgumentParser(description="signed (policy, table, dose, knob) gate")
    for f in ("--manifest", "--run", "--ckpt", "--table", "--dose"): ap.add_argument(f, required=True)
    a = ap.parse_args(argv)
    rc, msg, flag = check(a.manifest, a.run, a.ckpt, a.table, a.dose)
    print(msg); print("KNOBFLAG=%s" % flag)
    return rc

if __name__ == "__main__":
    sys.exit(main())
